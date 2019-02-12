""" Base classes and functions for ruleset classifiers """

import math
import operator as op
from functools import reduce
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
        if type(other)!=Rule:
            raise TypeError(f'{self} __eq__ {other}: a Ruleset can only be compared with another Ruleset')
        for r in self.rules:
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

    def count_conds(self):
        return sum([len(r.conds) for r in self.rules])

    def _set_possible_conds(self, pos_df, neg_df):
        """ Stores a list of all possible conds. """

        #Used in Rule::successors so as not to rebuild it each time,
        # and in exceptions_dl calculations because nCr portion of formula already accounts for no replacement.)

        self.possible_conds = []
        for feat in pos_df.columns.values:
            for val in set(pos_df[feat].unique()).intersection(set(neg_df[feat].unique())):
                self.possible_conds.append(Cond(feat, val))

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
            for i, p in zip(X_df.index,predictions):
                example = X_df[X_df.index==i]
                example_reasons = [rule for rule in self.rules if len(rule.covers(example))==1] if p else []
                reasons.append(example_reasons)
            return (predictions, reasons)

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
        if type(cond)==Cond:
            return Rule(self.conds+[cond])
        else:
            raise TypeError(f'{self} + {cond}: Rule objects can only conjoin Cond objects.')

    def __eq__(self, other):
        if type(other)!=Rule:
            raise TypeError(f'{self} __eq__ {other}: a Rule can only be compared with another rule')
        if len(self.conds)!=len(other.conds): return False
        return set([str(cond) for cond in self.conds]) == set([str(cond) for cond in other.conds])

    def isempty(self):
        return len(self.conds)==0

    def covers(self, df):
        """ Returns instances covered by the Rule. """
        covered = df.copy()
        for cond in self.conds:
            covered = cond.covers(covered)
        return covered

    def num_covered(self, df):
        return len(self.covers(df))

    def covered_feats(self):
        """ Returns list of features covered by the Rule """
        return [cond.feature for cond in self.conds]

    def grow(self, pos_df, neg_df, possible_conds, initial_rule=None, verbosity=0):
        """ Fit a new rule to add to a ruleset """

        if initial_rule is None:
            rule0 = Rule()
        else:
            rule0 = copy.deepcopy(initial_rule)

        if verbosity>=4:
            print('growing rule')
            print(rule0)
        rule1 = copy.deepcopy(rule0)
        while len(rule0.covers(neg_df)) > 0 and rule1 is not None: # Stop refining rule if no negative examples remain
            rule1 = best_successor(rule0, possible_conds, pos_df, neg_df, verbosity=verbosity)
            if rule1 is not None:
                rule0 = rule1
                if verbosity>=4:
                    print(f'negs remaining {len(rule0.covers(neg_df))}')

        if not rule0.isempty():
            if verbosity>=3: print(f'grew rule: {rule0}')
        else:
            #warnings.warn(f"grew an empty rule {rule0} over {len(pos_df)} pos and {len(neg_df)} neg", RuntimeWarning)#, stacklevel=1, source=None)
            pass

        self = rule0

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
                    if feat not in self.covered_feats(): # Conds already in Rule and Conds that contradict Rule aren't valid successors
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

    def covers(self, df):
        """ Returns instances covered by the Cond (i.e. those which are not in contradiction with it). """
        return df[df[self.feature]==self.val]
        #return [(Xi, yi) for Xi, yi in zip(X, y) if Xi[self.feat_index  ]==self.val]
    def num_covered(self, df):
        return len(self.covers(df))

##########################
##### BASE FUNCTIONS #####
##########################

def grow_rule(pos_df, neg_df, possible_conds, initial_rule=Rule(), verbosity=0):
    """ Fit a new rule to add to a ruleset """
    # Possible optimization: remove data after each added cond?

    rule0 = copy.deepcopy(initial_rule)
    if verbosity>=4:
        print('growing rule')
        print(rule0)
    rule1 = copy.deepcopy(rule0)
    while len(rule0.covers(neg_df)) > 0 and rule1 is not None: # Stop refining rule if no negative examples remain
        rule1 = best_successor(rule0, possible_conds, pos_df, neg_df, verbosity=verbosity)
        #print(f'growing rule... {rule1}')
        if rule1 is not None:
            rule0 = rule1
            if verbosity>=4:
                print(f'negs remaining {len(rule0.covers(neg_df))}')

    if not rule0.isempty():
        if verbosity>=2: print(f'grew rule: {rule0}')
        return rule0
    else:
        #warnings.warn(f"grew an empty rule {rule0} over {len(pos_df)} pos and {len(neg_df)} neg", RuntimeWarning)#, stacklevel=1, source=None)
        return rule0

def prune_rule(rule, prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=None, verbosity=0):
    """ Returns a pruned version of the Rule by removing Conds

        rule: Rule to prune
        prune_metric: function that returns value to maximize
        pos_pruneset: df of positive class examples
        neg_pruneset: df of non-positive class examples

        eval_index_on_ruleset (optional): tuple(rule_index, ruleset)
            pass the rest of the Rule's Ruleset (excluding the Rule in question),
            in order to prune the rule based on the performance of its entire Ruleset,
            rather than on the rule alone. For use during optimization stage.
    """

    if rule.isempty():
        warnings.warn(f"can't prune empty rule {rule}", RuntimeWarning)#, stacklevel=1, source=None)
        return rule

    if not eval_index_on_ruleset:

        # Currently-best pruned rule and its prune value
        best_rule = copy.deepcopy(rule)
        best_v = 0

        # Iterative test rule
        current_rule = copy.deepcopy(rule)

        while current_rule.conds:
            v = prune_metric(current_rule, pos_pruneset, neg_pruneset)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
            current_rule.conds.pop(-1)

        if verbosity>=2:
            if len(best_rule.conds)!=len(rule.conds):
                print(f'pruned rule: {best_rule}')
            else:
                print(f'pruned rule unchanged')
        return best_rule

    else:
        # Check if index matches rule to prune
        rule_index, ruleset = eval_index_on_ruleset
        if ruleset.rules[rule_index] != rule:
            raise ValueError(f'rule mismatch: {rule} - {ruleset.rules[rule_index]} in {ruleset}')

        current_ruleset = copy.deepcopy(ruleset)
        current_rule = current_ruleset.rules[rule_index]
        best_ruleset = copy.deepcopy(current_ruleset)
        best_v = 0

        # Iteratively prune and test rule over ruleset.
        # This is unfortunately expensive.
        while current_rule.conds:
            v = prune_metric(current_ruleset, pos_pruneset, neg_pruneset)
            if verbosity>=5: print(f'prune value of {current_rule}: {rnd(v)}')
            if v is None:
                return None
            if v >= best_v:
                best_v = v
                best_rule = copy.deepcopy(current_rule)
                best_ruleset = copy.deepcopy(current_ruleset)
            current_rule.conds.pop(-1)
            current_ruleset.rules[rule_index] = current_rule
        return best_rule

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

    ###################
    ##### METRICS #####
    ###################

def gain(self, other, pos_df, neg_df):
    """ Returns the information gain from self to other """

    p0count = self.num_covered(pos_df)
    p1count = other.num_covered(pos_df)
    n0count = self.num_covered(neg_df)
    n1count = other.num_covered(neg_df)
    return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

def precision(object, pos_df, neg_df):
    """ Returns precision value of object's classification.
        object: Cond, Rule, or Ruleset
    """

    pos_covered = object.covers(pos_df)
    neg_covered = object.covers(neg_df)
    total_n_covered = len(pos_covered)+len(neg_covered)
    if total_n_covered == 0:
        return None
    else:
        return len(pos_covered) / total_n_covered

def score_accuracy(predictions, actuals):
    """ For evaluating trained model on test set.

        predictions: <iterable<bool>> True for predicted positive class, False otherwise
        actuals:     <iterable<bool>> True for actual positive class, False otherwise
    """
    t = [pr for pr,act in zip(predictions,actuals) if pr==act]
    n = predictions
    return len(t)/len(n)

def accuracy(object, pos_pruneset, neg_pruneset):
    """ Returns accuracy value of object's classification.
        object: Cond, Rule, or Ruleset
    """
    P = len(pos_pruneset)
    N = len(neg_pruneset)
    if P + N == 0:
        return None

    tp = len(object.covers(pos_pruneset))
    tn = N - len(object.covers(neg_pruneset))
    return (tp + tn) / (P + N)

def best_successor(rule, possible_conds, pos_df, neg_df, verbosity=0):
    """ Returns for a Rule its best successor Rule according to FOIL information gain metric.

        eval_on_ruleset: option to evaluate gain with extra disjoined rules (for use with RIPPER's post-optimization)
    """

    best_gain = 0
    best_successor_rule = None

    for successor in rule.successors(possible_conds, pos_df, neg_df):
        g = gain(rule, successor, pos_df, neg_df)
        if g > best_gain:
            best_gain = g
            best_successor_rule = successor
    if verbosity>=5: print(f'gain {rnd(best_gain)} {best_successor_rule}')
    return best_successor_rule

    ###################
    ##### HELPERS #####
    ###################

def pos_neg_split(df, class_feat, pos_class):
    """ Split df into pos and neg classes. """
    pos_df = pos(df, class_feat, pos_class)
    neg_df = neg(df, class_feat, pos_class)
    return pos_df, neg_df

def df_shuffled_split(df, split_size, random_state=None):
    """ Returns tuple of shuffled and split DataFrame.
        split_size: proportion of rows to include in tuple[0]
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_at = int(len(df)*split_size)
    return (df[:split_at], df[split_at:])

def pos(df, class_feat, pos_class):
    """ Returns subset of instances that are labeled positive. """
    #""" Returns X,y subset that are labeled positive """
    return df[df[class_feat] == pos_class]
    #return [(Xi, yi) for Xi, yi in zip(X, y) if y==pos_class]

def neg(df, class_feat, pos_class):
    """ Returns subset of instances that are NOT labeled positive. """
    #""" Returns X,y subset that are NOT labeled positive """
    return df[df[class_feat] != pos_class]
    #return [(Xi, yi) for Xi, yi in zip(X, y) if y!=pos_class]

def num_pos(df, class_feat, pos_class):
    """ Returns number of instances that are labeled positive. """
    #""" Returns X,y subset that are labeled positive """
    return len(df[df[class_feat] == pos_class])
    #return len(_pos(X, y, pos_class))

def num_neg(df, class_feat, pos_class):
    """ Returns number of instances that are NOT labeled positive. """
    #""" Returns X,y subset that are NOT labeled positive """
    return len(df[df[class_feat] != pos_class])
    #return len(_neg(X, y, pos_class))

def nCr(n, r):
    """ Returns number of combinations C(n, r) """
    def product(numbers):
        return reduce(op.mul, numbers, 1)

    num = product(range(n, n-r, -1))
    den = product(range(1, r+1))
    return num//den

def rnd(float, places='default'):
    """ places: number of decimal places to round.
                set to 'default': defaults to 1 decimal place if float < 100, otherwise defaults to 0 places
    """
    if places=='default':
        if float<1:
            places = 2
        elif float<100:
            places = 1
        else:
            places = 0
    rounded = round(float, places)
    if rounded!=int(rounded):
        return rounded
    else:
        return int(rounded)

def argmin(list_):
    """ Returns index of minimum value. """
    lowest_val = list_[0]
    lowest_i = 0
    for i, val in enumerate(list_):
        if val < lowest_val:
            lowest_val = val
            lowest_i = i
    return lowest_i

def i_replaced(list_, i, value):
    """ Returns a new list with element i replaced by value.
        Pass None to value to return list with element i removed.
    """
    if value is not None:
        return list_[:i]+[value]+list_[i+1:]
    else:
        return list_[:i]+list_[i+1:]

def rm_covered(object, pos_df, neg_df):
    """ Return pos and neg dfs of examples that are not covered by object """
    return (pos_df.drop(object.covers(pos_df).index, axis=0, inplace=False),\
            neg_df.drop(object.covers(neg_df).index, axis=0, inplace=False))

def trainset_classfeat_posclass(df, y=None, class_feat=None, pos_class=None):
    """ Process params into trainset, class feature name, and pos class, for use in .fit methods. """

    # Ensure class feature is provided
    if y is None and class_feat is None:
        raise ValueError('y or class_feat argument is required')

    # Ensure no class feature name mismatch
    if y is not None and class_feat is not None \
            and hasattr(y, 'name') \
            and y.name != class_feat:
        raise ValueError(f'Value mismatch between params y {y.name} and class_feat {class_feat}. Besides, you only need to provide one of them.')

    # Set class feature name
    if class_feat is not None:
        # (IOW, pass)
        class_feat = class_feat
    elif y is not None and hasattr(y, 'name'):
        # If y is a pandas Series, try to get its name
        class_feat = y.name
    else:
        # Create a name for it
        class_feat = 'Class'

    # If necessary, merge y into df
    if y is not None:
        df[class_feat] = y

    # If provided, define positive class name. Otherwise, assign one.
    if pos_class is not None:
        pos_class = pos_class
    else:
        pos_class = df.iloc[0][class_feat]

    return (df, class_feat, pos_class)

########################################
##### BONUS: FUNCTIONS FOR BINNING #####
########################################

def find_numeric_feats(df, min_unique=10, ignore_feats=[]):
    """ Returns df features that seem to be numeric """
    feats = df.dtypes[(df.dtypes=='float64') | (df.dtypes=='int64')].index.tolist()
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

        if verbosity>=4: print (f'{feat}: fitting {len(df[feat].unique())} unique vals in {n_bins} bins')
        bin_ranges = []
        finish_i=-1
        sizes=[]
        for bin_i in range(0,n_bins):
            #print(f'bin {bin_i} of {range(n_bins)}')
            start_i = bin_size*(bin_i)
            finish_i = bin_size*(bin_i+1) if bin_i<n_bins-1 else len(sorted_df)-1
            while finish_i<len(sorted_df)-1 and finish_i!=0 and \
                    sorted_df.iloc[finish_i][feat]==sorted_df.iloc[finish_i-1][feat]: # ensure next bin begins on a new value
                finish_i+=1
                #print(f'{sorted_df.iloc[finish_i][feat]} {sorted_df.iloc[finish_i-1][feat]} {finish_i}')
            sizes.append(finish_i-start_i)
            #print(f'bin_i {bin_i}, start_i {start_i} {sorted_df.iloc[start_i][feat]}, finish_i {finish_i} {sorted_df.iloc[finish_i][feat]}')
            start_val = sorted_values[start_i]
            finish_val = sorted_values[finish_i]
            bin_range = (start_val, finish_val)
            bin_ranges.append(bin_range)
        if verbosity>=5: print(f'-bin sizes {sizes}; dataVMR={rnd(var(df[feat])/mean(df[feat]))}, binVMR={rnd(var(sizes)/mean(sizes))}')#, axis=None, dtype=None, out=None, ddof=0)})
        return bin_ranges

    # Create dict to store fit definitions for each feature
    fit_dict = {}
    feats_to_fit = find_numeric_feats(df,ignore_feats=ignore_feats)
    if verbosity==2: print(f'fitting bins for features {feats_to_fit}')
    if verbosity>=2:
        print()

    # Collect fits in dict
    count=1
    for feat in feats_to_fit:
        fit = bin_fit_feat(df, feat, n_bins=n_bins)
        fit_dict[feat] = fit
    return fit_dict

def bin_transform(df, fit_dict, names_precision=2):
    """
    Uses a pre-collected dictionary of fits to transform df features into bins
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
            min = bin_fit_list[0][0]
            max = bin_fit_list[-1][1]
            for bin_fit in bin_fits:
                if value <= bin_fit[1]:
                    start_name = str(round(bin_fit[0], names_precision))
                    finish_name = str(round(bin_fit[1], names_precision))
                    bin_name = '-'.join([start_name, finish_name])
                    return bin_name
            if value < min:
                return min
            elif value > max:
                return max
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
