""" Base classes and functions for ruleset classifiers """

import math
import numpy as np
import copy
import warnings
from numpy import var, mean
import time

class Ruleset:
    """ Base Ruleset model.
        Implements collection of Rules in disjunctive normal form.
    """

    def __init__(self, rules=None):
        if rules is None:
            self.rules = []
        else:
            self.rules = rules
        self.cond_count = 0

    def __str__(self):
        return ' V '.join([str(rule) for rule in self.rules])

    def __repr__(self):
        ruleset_str = self.__str__()
        return f'<Ruleset object: {ruleset_str}>'

    def __getitem__(self, index):
        return self.rules[index]

    def __len__(self):
        return len(self.rules)

    def truncstr(self, limit=2, direction='left'):
        """ Return Ruleset string representation limited to a specified number of rules.

            limit: how many rules to return
            direction: which part to return. (valid options: 'left', 'right')
        """
        if len(self.rules)>=limit:
            if direction=='left':
                return Ruleset(self.rules[:limit]).__str__()+'...'
            elif direction=='right':
                return '...'+Ruleset(self.rules[-limit:]).__str__()
            else:
                raise ValueError('direction param must be "left" or "right"')
        else:
            return self.__str__()

    def __eq__(self, other):
        #if type(other)!=Ruleset:
        #    raise TypeError(f'{self} __eq__ {other}: a Ruleset can only be compared with another Ruleset')
        for r in self.rules: # TODO: Ideally, should implement a hash function--in practice speedup would be insignificant
            if r not in other.rules: return False
        for r in other.rules: # Check the other way around too. (Can't compare lengths instead b/c there might be duplicate rules.)
            if r not in self.rules: return False
        return True

    def out_pretty(self):
        """ Prints Ruleset line-by-line. """
        ruleset_str = str([str(rule) for rule in self.rules]).replace(' ','').replace(',',' V\n').replace("'","")
        print(ruleset_str)

    def copy(self, n_rules_limit=None):
        """ Returns a deep copy of self.

            n_rules_limit (optional): keep only this many rules from the original.
        """
        result = copy.deepcopy(self)
        if n_rules_limit is not None:
            result.rules = result.rules[:n_rules_limit]
        return result

    def covers(self, df):
        """ Returns instances covered by the Ruleset. """

        if not self.rules:
            return df
        else:
            covered = self.rules[0].covers(df).copy()
            for rule in self.rules[1:]:
                covered = covered.append(rule.covers(df))
            covered = covered.drop_duplicates()
            return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def add(self, rule):
        self.rules.append(rule)

    def count_rules(self):
        """ Returns number of rules in the Ruleset.
            (For ease of use for users who don't make a habit of directly accessing the Ruleset object.)
        """
        return len(self.rules)

    def count_conds(self):
        """ Returns the total number of conditions in the Ruleset. This is a measurement of complexity that's the conceptual equivalent of counting the nodes in a decision tree. """
        return sum([len(r.conds) for r in self.rules])

    def _set_possible_conds(self, pos_df, neg_df):
        """ Stores a list of all possible conds. """

        #Used in Rule::successors so as not to rebuild it each time,
        # and in exceptions_dl calculations because nCr portion of formula already accounts for no replacement.)

        self.possible_conds = []
        for feat in pos_df.columns.values:
            for val in set(pos_df[feat].unique()).intersection(set(neg_df[feat].unique())):
                self.possible_conds.append(Cond(feat, val))

    def trim_conds(self, max_total_conds=None):
        """ Reduce the total number of Conds in a Ruleset by removing Rules """
        if max_total_conds is not None:
            while len(self.rules) > 0 and self.count_conds() > max_total_conds:
                self.rules.pop(-1)

    def predict(self, X_df, give_reasons=False):
        """ Predict classes of data using a fit Ruleset model.

            args:
                X_df <DataFrame>: examples to make predictions on.

                give_reasons (optional) <bool>: whether to provide reasons for each prediction made.

            returns:
                list of <bool> values corresponding to examples. True indicates positive predicted class; False non-positive class.

                If give_reasons is True, returns a tuple that contains the above list of predictions
                    and a list of the corresponding reasons for each prediction;
                    for each positive prediction, gives a list of all the covering Rules, for negative predictions, an empty list.
        """

        covered_indices = set(self.covers(X_df).index.tolist())
        predictions = [i in covered_indices for i in X_df.index]

        if not give_reasons:
            return predictions
        else:
            reasons = []
            # For each Ruleset-covered example, collect list of every Rule that covers it;
            # for non-covered examples, collect an empty list
            for i, p in zip(X_df.index, predictions):
                example = X_df[X_df.index==i]
                example_reasons = [rule for rule in self.rules if len(rule.covers(example))==1] if p else []
                reasons.append(example_reasons)
            return (predictions, reasons)

    def predict_proba(self, X_df, give_reasons=False, ret_n=False, min_samples=1, discretize=True, bin_transformer=None):
        """ Predict probabilities for each class using a fit Ruleset model.

                args:
                    X_df <DataFrame>: examples to make predictions on.

                    give_reasons (optional) <bool>: whether to provide reasons for each prediction made.
                    min_samples (optional) <int>: return None for each example proba that lack this many samples
                                                  set to None to ignore. (default=None)

                    give_reasons (optional) <bool>: whether to also return reasons for each prediction made.
                    ret_n (optional) <bool>: whether to also return the number of samples used for calculating each examples proba

                returns:
                    numpy array of values corresponding to each example's classes probabilities, or, if give_reasons or ret_n, a tuple containing proba array and list(s) of desired returns

        """

        # Apply binning if necessary
        if discretize and bin_transformer is not None:
            df = X_df.copy()
            df = bin_transform(df, bin_transformer)
        else:
            df = X_df

        # probas for all negative predictions
        uncovered_proba = weighted_avg_freqs([self.uncovered_class_freqs])
        uncovered_n = sum(self.uncovered_class_freqs)

        # make predictions
        predictions, covering_rules = self.predict(df, give_reasons=True)
        N = []

        # collect probas
        probas = np.empty(shape=(len(predictions),uncovered_proba.shape[0]))
        for i, (p, cr) in enumerate(zip(predictions, covering_rules)):
            n = sum([sum(rule.class_freqs) for rule in cr]) # if user requests, check to ensure valid sample size
            if p and (n < 1 or (min_samples and n < min_samples)):
                probas[i, :] = None
                N.append(n)
            elif (not p) and (uncovered_n < 1 or uncovered_n < min_samples):
                probas[i, :] = None
                N.append(n)
            elif p: # pos prediction
                probas[i, :] = weighted_avg_freqs([rule.class_freqs for rule in cr])
                N.append(n)
            else: # neg prediction
                probas[i, :] = uncovered_proba
                N.append(uncovered_n)

        # return probas (and optional extras)
        result = flagged_return([True, give_reasons, ret_n], [probas, covering_rules, N])
        return result

class Rule:
    """ Class implementing conjunctions of Conds """

    def __init__(self, conds=None):
        if conds is None:
            self.conds = []
        else:
            self.conds = conds

    def __str__(self):
        rule_str = str([str(cond) for cond in self.conds]).replace(',','^').replace("'","").replace(' ','')
        return rule_str

    def __repr__(self):
        rule_str = str([str(cond) for cond in self.conds]).replace(', ','^').replace("'","").replace(' ','')
        return f'<Rule object: {rule_str}>'

    def __add__(self, cond):
        if isinstance(cond, Cond):
            return Rule(self.conds+[cond])
        else:
            raise TypeError(f'{self} + {cond}: Rule objects can only conjoin Cond objects.')

    def __eq__(self, other):
        #if type(other)!=Rule:
        #    raise TypeError(f'{self} __eq__ {other}: a Rule can only be compared with another rule')
        if len(self.conds)!=len(other.conds): return False
        return set([str(cond) for cond in self.conds]) == set([str(cond) for cond in other.conds])

    def isempty(self):
        return len(self.conds)==0

    def covers(self, df):
        """ Returns instances covered by the Rule. """
        covered = df.head(len(df))
        for cond in self.conds:
            covered = cond.covers(covered)
        return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def covered_feats(self):
        """ Returns list of features covered by the Rule """
        return [cond.feature for cond in self.conds]

    #############################################
    ##### Rule::grow/prune helper functions #####
    #############################################

    def successors(self, possible_conds, pos_df, neg_df):
        """ Returns a list of all valid successor rules.

        possible_conds: list of Conds to consider conjoining to create successors.
                        passing None defaults to create this param from pos_df and neg_df --
                        however, if pos_df and neg_df are data subsets, it will only generate possible_conds
                        from their available values.
        """

        if possible_conds is not None:
            successor_conds = [cond for cond in possible_conds if cond not in self.conds]
            return [Rule(self.conds+[cond]) for cond in successor_conds]
        else:
            successor_rules = []
            for feat in pos_df.columns.values:
                for val in set(pos_df[feat].unique()).intersection(set(neg_df[feat].unique())):
                    if feat not in self.covered_feats(): # Conds already in Rule and Conds that contradict Rule aren't valid successors / NB Rules are short; this is unlikely to be worth the overhead of cheacking
                        successor_rules.append(self+Cond(feat, val))
            return successor_rules

class Cond:
    """ Class implementing conditional. """

    def __init__(self, feature, val):
        self.feature = feature
        self.val = val

    def __str__(self):
        return f'{self.feature}={self.val}'

    def __repr__(self):
        return f'<Cond object: {self.feature}={self.val}>'

    def __eq__(self, other):
        return self.feature == other.feature and self.val == other.val

    def __hash__(self):
        return hash((self.feature, self.val))

    def covers(self, df):
        """ Returns instances covered by the Cond (i.e. those which are not in contradiction with it). """
        return df[df[self.feature]==self.val]

    def num_covered(self, df):
        return len(self.covers(df))

class Timer:
    """ Simple, useful class for keeping track of how long something has been taking """

    def __init__(self):
        """ Create Timer object and hit start. """

        self.start = time.time()

    def buzz(self, reset=True):
        """ Returns time elapsed since Timer was created or reset, in seconds.

            args:
                reset (optional): whether to reset the clock.
        """

        last_buzz = self.start
        now = time.time()
        if reset:
            self.start = now
        return(str(int(now-last_buzz)))

    def stop(self):
        """ Freeze the clock at the amount of time elapsed since Timer was created or reset. """

        self.elapsed = time.time()-self.start

########################################
##### BONUS: FUNCTIONS FOR BINNING #####
########################################

def bin_df(df, n_discretize_bins=10, ignore_feats=[], verbosity=0):
    """ Returns df with seemingly numeric features binned, and the bin_transformer, if binning takes places. """

    isbinned = False
    min_unique = n_discretize_bins if n_discretize_bins else 10 # Minimum unique values to consider continuous. Override with 10 (rather than halt) to allow warnings to occur
    numeric_feats = find_numeric_feats(df, min_unique=min_unique, ignore_feats=ignore_feats)

    if numeric_feats:
        if n_discretize_bins:
            if verbosity==1:
                print(f'binning data...\n')
            elif verbosity>=2:
                print(f'binning features {numeric_feats}...')
            binned_df = df.copy()
            bin_transformer = fit_bins(binned_df, n_bins=n_discretize_bins, output=False, ignore_feats=ignore_feats, verbosity=verbosity)
            binned_df = bin_transform(binned_df, bin_transformer)
            isbinned = True
        else:
            n_unique_values = sum([len(u) for u in [df[f].unique() for f in numeric_feats]])
            warnings.warn(f"There are {len(numeric_feats)} apparent numeric features: {numeric_feats}. \n Treating {n_unique_values} numeric values as nominal. To auto-discretize features, assign a value to parameter 'n_discretize_bins.'", RuntimeWarning)

    if isbinned:
        return binned_df, bin_transformer
    else:
        return df, None

def find_numeric_feats(df, min_unique=10, ignore_feats=[]):
    """ Returns df features that seem to be numeric. """
    #feats = df.dtypes[(df.dtypes=='float64') | (df.dtypes=='int64')].index.tolist()
    feats = df._get_numeric_data().columns
    feats = [f for f in feats if f not in ignore_feats]
    feats = [f for f in feats if len(df[f].unique())>min_unique]
    return feats

def fit_bins(df, n_bins=5, output=False, ignore_feats=[], verbosity=0):
    """
    Returns a dict definings fits for numerical features
    A fit is an ordered list of tuples defining each bin's range

    Returned dict allows for fitting to training data and applying the same fit to test data
    to avoid information leak.
    """

    def bin_fit_feat(df, feat, n_bins=10):
        """
        Returns list of tuples defining bin ranges for a numerical feature
        """
        n_bins = min(n_bins, len(df[feat].unique())) # In case there are fewer unique values than n_bins
        bin_size = len(df)//n_bins
        sorted_df = df.sort_values(by=[feat])
        sorted_values = sorted_df[feat].tolist()

        if verbosity>=4: print (f'{feat}: fitting {len(df[feat].unique())} unique vals into {n_bins} bins')
        bin_ranges = []
        finish_i=-1
        sizes=[]

        bin = 0
        start_i = 0
        while bin < n_bins and start_i < len(sorted_values):
            finish_i = min(start_i+bin_size, len(sorted_df)-1)
            while finish_i<len(sorted_df)-1 and finish_i!=0 and \
                    sorted_df.iloc[finish_i][feat]==sorted_df.iloc[finish_i-1][feat]: # ensure next bin begins on a new value
                finish_i+=1
                #print(f'{sorted_df.iloc[finish_i][feat]} {sorted_df.iloc[finish_i-1][feat]} {finish_i}')
            sizes.append(finish_i-start_i)
            if verbosity >=4: print(f'bin #{bin}, start_idx {start_i} value: {sorted_df.iloc[start_i][feat]}, finish_idxx {finish_i} value: {sorted_df.iloc[finish_i][feat]}')
            start_val = sorted_values[start_i]
            finish_val = sorted_values[finish_i]
            bin_range = (start_val, finish_val)
            bin_ranges.append(bin_range)

            start_i = finish_i+1
            bin += 1

        if verbosity>=4: print(f'-bin sizes {sizes}; dataVMR={rnd(var(df[feat])/mean(df[feat]))}, binVMR={rnd(var(sizes)/mean(sizes))}')#, axis=None, dtype=None, out=None, ddof=0)})
        return bin_ranges

    # Create dict to store fit definitions for each feature
    fit_dict = {}
    feats_to_fit = find_numeric_feats(df,ignore_feats=ignore_feats)
    if verbosity==2: print(f'fitting bins for features {feats_to_fit}')
    if verbosity>=2: print()

    # Collect fits in dict
    count=1
    for feat in feats_to_fit:
        fit = bin_fit_feat(df, feat, n_bins=n_bins)
        fit_dict[feat] = fit
    return fit_dict

def bin_transform(df, fit_dict, names_precision=2):
    """
    Uses a pre-collected dictionary of fits to transform df features into bins.
    Returns the fit df rather than modifying inplace.
    """

    def bin_transform_feat(df, feat, bin_fits, names_precision=names_precision):
        """
        Returns new dataframe with n_bin bins replacing each numerical feature
        """

        def renamed(bin_fit_list, value, names_precision=names_precision):
            """
            Returns bin string name for a given numberical value
            Assumes bin_fit_list is ordered
            """
            min_val = bin_fit_list[0][0]
            max_val = bin_fit_list[-1][1]
            for bin_fit in bin_fits:
                if value <= bin_fit[1]:
                    start_name = str(round(bin_fit[0], names_precision))
                    finish_name = str(round(bin_fit[1], names_precision))
                    bin_name = '-'.join([start_name, finish_name])
                    return bin_name
            if value < min_val:
                return min_val
            elif value > max_val:
                return max_val
            else:
                raise ValueError('No bin found for value', value)

        renamed_values = []
        for value in df[feat]:
            bin_name = renamed(bin_fits, value, names_precision)
            renamed_values.append(bin_name)

        return renamed_values

    # Replace each feature with bin transformations
    for feat, bin_fits in fit_dict.items():
        feat_transformation = bin_transform_feat(df, feat, bin_fits, names_precision=names_precision)
        df[feat] = feat_transformation
    return df

######################
######## MATH ########
######################

def weighted_avg_freqs(counts):
    """ Return weighted mean proportions of counts in the list

        counts <list<tuple>>
    """
    arr = np.array(counts)
    total = arr.flatten().sum()
    return arr.sum(axis=0) / total

def flagged_return(flags, objects):
    """ Returns only objects with corresponding True flags
        Useful when """
    if sum(flags)==1:
        return objects[0]
    elif sum(flags)>1:
        return tuple([object for flag, object in zip(flags, objects) if flag])
    else:
        return ()
