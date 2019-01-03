"""
This module implements ... (RIPPER) algorithm
for growing classification rulesets.
See https://www.let.rug.nl/nerbonne/teach/learning/cohen95fast.pdf
"""

# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd
import copy
import math
import warnings

import base
from base import Ruleset, Rule, Cond
from base import rnd, fit_bins, bin_transform

class RIPPER:
    """ Class for generating ruleset classification models. """

    def __init__(self, class_feat, pos_class=None, k=2, prune_size=.33, dl_allowance=64, verbosity=0):
        """ class_feat: column name of class feature

            pos_class (optional):    name of positive class. If not provided, will default to class of first training example
            k (optional):            number of RIPPERk optimization iterations
            prune_size (optional):   proportion of training set to be used for pruning.
            dl_allowance (optional): during Ruleset grow phase, this is the maximum Ruleset description length permitted
                                     beyond the minimum description length seen so far (measured in bits)
            verbosity (optional):
                       1: Results of each major stage
                       2: Ruleset grow/optimization steps
                       3: Ruleset grow/optimization calculations
                       4: Rule grow/prune steps
                       5: Rule grow/prune calculations
        """
        self.class_feat = class_feat
        self.pos_class = pos_class
        self.prune_size = prune_size
        self.dl_allowance = dl_allowance
        self.k = k
        self.verbosity = verbosity

    def __str__(self):
        fitstr = f'fit ruleset={self.ruleset_}' if hasattr(self,'ruleset_') else 'not fit'
        return f'<IREP object {fitstr}>'
    __repr__ = __str__

    def fit(self, df, n_discretize_bins=None, seed=None):
        """ Fit an IREP to data by growing a classification Ruleset in disjunctive normal form.
            class_feat: name of DataFrame class feature

            n_discretize_bins (optional): try to fit apparent numeric attributes into n_discretize_bins discrete bins.
            seed: (optional) random state for grow/prune split (if pruning)
        """

        # If not given by __init__, define positive class here
        if not self.pos_class:
            self.pos_class = df.iloc[0][class_feat]

        # Anything to discretize?
        numeric_feats = base.find_numeric_feats(df, min_unique=n_discretize_bins, ignore_feats=[self.class_feat])
        if numeric_feats:
            if n_discretize_bins is not None:
                if self.verbosity==1:
                    print(f'binning data...\n')
                elif self.verbosity>=2:
                    print(f'binning features {numeric_feats}...')
                self.bin_transformer_ = fit_bins(df, n_bins=n_discretize_bins, output=False, ignore_feats=[self.class_feat], verbosity=self.verbosity)
                #print(f'bin transformer {bin_transformer}')
                binned_df = bin_transform(df, self.bin_transformer_)
            else:
                n_unique_values = sum([len(u) for u in [df[f].unique() for f in numeric_feats]])
                warnings.warn(f'Optional param n_discretize_bins=None, but there are apparent numeric features: {numeric_feats}. \n Treating {n_unique_values} numeric values as nominal', RuntimeWarning)
                binned_df=None
        else:
            binned_df=None
        # Split df into pos, neg classes
        pos_df, neg_df = base.pos_neg_split(df, self.class_feat, self.pos_class) if binned_df is None else base.pos_neg_split(binned_df, self.class_feat, self.pos_class)
        pos_df = pos_df.drop(self.class_feat,axis=1)
        neg_df = neg_df.drop(self.class_feat,axis=1)

        # Grow Ruleset
        self.ruleset_ = Ruleset()
        self.ruleset_ = self.grow_ruleset(pos_df, neg_df,
            prune_size=self.prune_size, dl_allowance=self.dl_allowance,
            seed=seed)
        if self.verbosity >= 1:
            print()
            print('GREW INITIAL RULESET:')
            self.ruleset_.out_pretty()
            print()

        # Optimize Ruleset
        for iter in range(self.k):
            # Create new but reproducible random seed (if applicable)
            iter_seed = seed+100 if seed is not None else None
            # Run optimization iteration
            if self.verbosity>=1: print(f'optimization run {iter+1} of {self.k}')
            newset = self.optimize_ruleset(self.ruleset_, pos_df, neg_df, prune_size=self.prune_size, seed=iter_seed)

            if self.verbosity>=1:
                print()
                print('OPTIMIZED RULESET:')
                if self.verbosity>=2: print(f'iteration {iter+1} of {self.k}\n modified rules {[i for i in range(len(self.ruleset_.rules)) if self.ruleset_.rules[i]!= newset.rules[i]]}')
                newset.out_pretty()
                print()
            self.ruleset_ = newset

        # Cover any last remaining positives
        pos_remaining, neg_remaining = base.pos_neg_split(df, self.class_feat, self.pos_class)
        pos_remaining = pos_remaining.drop(self.class_feat,axis=1)
        neg_remaining = neg_remaining.drop(self.class_feat,axis=1)
        pos_remaining, neg_remaining = base.rm_covered(self.ruleset_, pos_remaining, neg_remaining)
        if len(pos_remaining)>=1:
            if self.verbosity>=2:
                print(f'{len(pos_remaining)} pos left. Growing final rules...')
            newset = self.grow_ruleset(pos_df, neg_df, initial_ruleset=self.ruleset_,
                prune_size=self.prune_size, dl_allowance=self.dl_allowance,
                seed=seed)
            if self.verbosity>=1:
                print('GREW FINAL RULES')
                newset.out_pretty()
                print()
            self.ruleset_ = newset
        else:
            if self.verbosity>=1: print('All pos covered\n')

        # Remove any rules that don't improve dl
        if self.verbosity>=2: print('Optimizing dl...')
        mdl_subset, _ = rs_total_bits(self.ruleset_, self.ruleset_.possible_conds, pos_df, neg_df,
                                        bestsubset_dl=True, ret_bestsubset=True, verbosity=self.verbosity)
        self.ruleset_ = mdl_subset
        if self.verbosity>=1:
            print('FINAL RULESET:')
            self.ruleset_.out_pretty()
            print()

    def predict(self, X):
        """ Predict classes of X """

        if hasattr(self, 'ruleset_'):
            covered_indices = self.ruleset_.covers(X).index
            return [covered_i in covered_indices for covered_i in X.index]
        else:
            raise AttributeError('You should fit an IREP object before making predictions with it.')

    def score(self, X, y, score_function):
        """ Test performance of fit Ruleset.
            score_function: function that takes parameters (actuals, predictions)
            Examples: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
        """

        predictions = self.predict(X)
        actuals = [yi==self.pos_class for yi in y]
        return score_function(actuals, predictions)

    def grow_ruleset(self, pos_df, neg_df, prune_size, dl_allowance, initial_ruleset=None, seed=None, verbose=False):
        """ Grow a Ruleset with pruning. """
        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()

        if initial_ruleset is None:
            ruleset = Ruleset()
            ruleset._set_possible_conds(pos_df, neg_df)
        else:
            ruleset = copy.deepcopy(initial_ruleset)

        ruleset_dl = None
        mdl = None      # Minimum encountered description length (in bits)
        dl_diff = 0
        if self.verbosity>=2:
            print('growing ruleset...')
            print()
        while len(pos_remaining) > 0 and dl_diff <= self.dl_allowance:
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, prune_size, seed=seed)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, prune_size, seed=seed)
            print(f'pos_growset {len(pos_growset)} pos_pruneset {len(pos_pruneset)}')
            print(f'neg_growset {len(neg_growset)} neg_pruneset {len(neg_pruneset)}')
            if len(pos_growset)==0: break
            #pos_growset, pos_pruneset, neg_growset, neg_pruneset = base.ensure_samples([pos_growset, pos_pruneset, neg_growset, neg_pruneset], min=1, tries=20)
            # Check to make sure at least one example of each type

            # Grow Rule
            grown_rule = base.grow_rule(pos_growset, neg_growset, ruleset.possible_conds, verbosity=self.verbosity)
            #if grown_rule.isempty():
            #    break
            if self.verbosity>=3: print(f'grew rule: {grown_rule}')

            # Prune Rule
            pruned_rule = base.prune_rule(grown_rule, growphase_prune_metric, pos_pruneset, neg_pruneset, verbosity=self.verbosity)

            # Add rule; calculate new description length
            ruleset.add(pruned_rule) # Unlike IREP, IREP*/RIPPER stopping condition is inclusive: "After each rule is added, the total description length of the rule set and examples is computed."
            if self.verbosity>=2:
                print(f"updated ruleset: {ruleset.truncstr(direction='right')}")
                print()

            if ruleset_dl is None:   # First Rule to be added
                rule_dl = r_theory_bits(pruned_rule, ruleset.possible_conds, verbosity=self.verbosity)
                theory_dl = rule_dl
                data_dl = exceptions_bits(ruleset, pos_df, neg_df, verbosity=self.verbosity)
                ruleset_dl = theory_dl + data_dl
                mdl = ruleset_dl
            else:
                rule_dl = r_theory_bits(pruned_rule, ruleset.possible_conds, verbosity=self.verbosity)
                theory_dl += rule_dl
                data_dl = exceptions_bits(ruleset, pos_df, neg_df, verbosity=self.verbosity)
                ruleset_dl = theory_dl + data_dl
                dl_diff = ruleset_dl - mdl

            if self.verbosity>=3:
                print(f'rule dl: {rnd(rule_dl)}')
                print(f'updated theory dl: {rnd(theory_dl)}')
                print(f'exceptions: {rnd(data_dl)}')
                print(f'total dl: {rnd(ruleset_dl)}')
                if dl_diff<=self.dl_allowance:
                    print(f'mdl {rnd(mdl)} (diff {rnd(dl_diff)} <= {rnd(self.dl_allowance)})')
                else:
                    print(f'mdl {rnd(mdl)} dl-halt: diff {rnd(dl_diff)} exceeds allowance ({rnd(self.dl_allowance)})')

            mdl = ruleset_dl if ruleset_dl<mdl else mdl

            # Remove covered examples
            pos_remaining, neg_remaining = base.rm_covered(pruned_rule, pos_remaining, neg_remaining)

            if self.verbosity>=3:
                print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                print()
        return ruleset

    def optimize_ruleset(self, ruleset, pos_df, neg_df, prune_size, seed=None):
        if self.verbosity>=2:
            print('optimizing ruleset...')
            print()

        pos_remaining = pos_df.copy()
        neg_remaining = neg_df.copy()
        original_ruleset = copy.deepcopy(ruleset)
        if self.verbosity>=4: print('calculate original ruleset potential dl...')
        original_dl = rs_total_bits(original_ruleset, original_ruleset.possible_conds, pos_df, neg_df, bestsubset_dl=True, verbosity=self.verbosity)
        if self.verbosity>=3:
            print(f'original ruleset potential dl: {rnd(original_dl)}')
            print()
        new_ruleset = copy.deepcopy(ruleset)
        #new_ruleset = original_ruleset.copy(0)

        for i, rule in enumerate(original_ruleset.rules):
            pos_growset, pos_pruneset = base.df_shuffled_split(pos_remaining, prune_size, seed=seed)
            neg_growset, neg_pruneset = base.df_shuffled_split(neg_remaining, prune_size, seed=seed)

            #disjs = Ruleset(base.i_replaced(ruleset.rules, i, None))  # the Ruleset without the current rule

            # Create alternative rules
            if self.verbosity>=4: print(f'creating replacement for {i} {ruleset.rules[i]}')
            g_replacement = base.grow_rule(pos_growset, neg_growset, original_ruleset.possible_conds, initial_rule=Rule(), verbosity=self.verbosity)
            replacement_ruleset = Ruleset(base.i_replaced(original_ruleset.rules, i, g_replacement))
            pr_replacement = base.prune_rule(g_replacement, optimization_prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=(i,replacement_ruleset), verbosity=self.verbosity)
            replacement_ruleset = Ruleset(base.i_replaced(original_ruleset.rules, i, pr_replacement))
            if self.verbosity>=3:
                print(f'grew replacement {g_replacement}')
                print(f'pruned replacement is {pr_replacement}')

            if self.verbosity>=3: print(f'creating revision for {i} {ruleset.rules[i]}')
            g_revision = base.grow_rule(pos_growset, neg_growset, original_ruleset.possible_conds, initial_rule=ruleset.rules[i], verbosity=self.verbosity)
            revision_ruleset = Ruleset(base.i_replaced(original_ruleset.rules, i, g_revision))
            pr_revision = base.prune_rule(g_revision, optimization_prune_metric, pos_pruneset, neg_pruneset, eval_index_on_ruleset=(i,revision_ruleset), verbosity=self.verbosity)
            revision_ruleset = Ruleset(base.i_replaced(original_ruleset.rules, i, pr_revision))
            if self.verbosity>=3:
                print(f'grew revision {g_replacement}')
                print(f'pruned revision is {pr_replacement}')
                print()

            # Calculate alternative Rulesets' respective lowest potential dls to identify the best version
            if self.verbosity>=3: print(f'calculate potential dl for ds with replacement {pr_replacement}')
            replacement_dl = rs_total_bits(replacement_ruleset, original_ruleset.possible_conds, pos_df, neg_df, bestsubset_dl=True, verbosity=self.verbosity)\
                             if pr_replacement!=rule else original_dl
            if self.verbosity>=3: print(f'calculate potential dl for ds with revision {pr_revision}')
            revision_dl = rs_total_bits(revision_ruleset, original_ruleset.possible_conds, pos_df, neg_df, bestsubset_dl=True, verbosity=self.verbosity)\
                          if pr_revision!=rule else original_dl
            best_rule = [rule, pr_replacement, pr_revision][base.argmin([original_dl, replacement_dl, revision_dl])]

            if self.verbosity>=2:
                print(f'\nrule {i+1} of {len(original_ruleset.rules)}')
                rep_str = pr_replacement.__str__() if pr_replacement!=rule else 'unchanged'
                rev_str = pr_revision.__str__() if pr_revision!=rule else 'unchanged'
                best_str = best_rule.__str__() if best_rule!=rule else 'unchanged'
                if self.verbosity==2:
                    print(f'original: {rule}')
                    print(f'replacement: {rep_str}')
                    print(f'revision: {rev_str}')
                    print(f'*best: {best_str}')
                    print()
                else:
                    print(f'original: {rule}) | {rnd(original_dl)} bits')
                    print(f'replacement: {rep_str} | {rnd(replacement_dl)} bits')
                    print(f'revision: {rev_str} | {rnd(revision_dl)} bits')
                    print(f'*best: {best_str} | {rnd(min([replacement_dl, revision_dl, original_dl]))} bits')
                    print()
            new_ruleset.rules[i] = best_rule
            #new_ruleset.add(best_rule)

            # Remove covered examples
            pos_remaining, neg_remaining = base.rm_covered(rule, pos_remaining, neg_remaining)
            if self.verbosity>=3:
                print(f'examples remaining: {len(pos_remaining)} pos, {len(neg_remaining)} neg')
                print()

            # If there are no pos data remaining to train optimization (possible where optimization run >1), keep remaining rules the same
            if len(pos_remaining)==0: break

        return new_ruleset

###################################
##### RIPPER-specific Metrics #####
###################################

def growphase_prune_metric(rule, pos_pruneset, neg_pruneset):
    """ RIPPER/IREP* prune metric.
        Returns the prune value of a candidate Rule.

        Cohen's formula is (p-n) / (p+n).
        Unclear from the paper how they handle divzero (where p+n=0), so I Laplaced it.
        Weka's solution was to modify the formula to (p+1)/(p+n+2), but the (non-NaN) values I got appeared closer to those of the original formula.
    """
    p = rule.num_covered(pos_pruneset)
    n = rule.num_covered(neg_pruneset)
    return (p - n + 1) / (p + n + 1)

def optimization_prune_metric(rule, pos_pruneset, neg_pruneset):
    return base.accuracy(rule, pos_pruneset, neg_pruneset)

def r_theory_bits(rule, possible_conds, verbosity=0):
    if type(rule) != Rule:
        raise TypeError(f'param rule in r_theory_bits should be type Rule')
    k = len(rule.conds)                                 # Number of rule conditions
    n = len(possible_conds)          # Number of possible conditions
    pr = k/n                                            #

    S = k*math.log2(1/pr) + (n-k)*math.log2((1/(1-pr)))  # S(n, k, pr)
    K = math.log2(k)                                     # Number bits need to send integer k
    rule_dl = 0.5*(K + S)                                # Divide by 2 a la Quinlan. Cohen: "to adjust for possible redundency in attributes"
    if verbosity>=5: print(f'rule theory bits| {rule} k {k} n {n} pr {rnd(pr)}: {rnd(rule_dl)} bits')

    return rule_dl

def rs_theory_bits(ruleset, possible_conds, pos_df, neg_df, verbosity=0):
    if type(ruleset) != Ruleset:
        raise TypeError(f'param ruleset in rs_theory_bits should be type Ruleset')
    """ Returns sum of theory bits for each Rule in ruleset """
    #rem_pos = copy.deepcopy(pos_df)
    #rem_neg = copy.deepcopy(neg_df)
    total = 0
    for rule in ruleset.rules:
        total += r_theory_bits(rule, possible_conds, verbosity=verbosity)
        #total += rule_bits(rule, possible_conds, rem_pos, rem_neg, verbosity=verbosity)
        #rem_pos, rem_neg = base.rm_covered(rule, rem_pos, rem_neg)
    if verbosity>=5: print(f'ruleset theory bits| {rnd(total)}')
    return total

def exceptions_bits(ruleset, pos_df, neg_df, verbosity=0):
    if type(ruleset) != Ruleset:
        raise TypeError(f'to avoid double-counting, exceptions_bits should calculate exceptions over entire set of rules with type Ruleset')
    N = len(pos_df) + len(neg_df)                           # Total number of examples
    p = ruleset.num_covered(pos_df) + ruleset.num_covered(neg_df) # Total number of examples classified as positive = total covered
    fp = ruleset.num_covered(neg_df)                           # Number false positives = negatives covered by the ruleset
    fn = len(pos_df) - ruleset.num_covered(pos_df)             # Number false negatives = positives not covered by the ruleset
    exceptions_dl = math.log2(base.nCr(p,fp)) + math.log2(base.nCr((N-p),fn))
    if verbosity>=5: print(f'exceptions_bits| {ruleset.truncstr()}: \n N {N} p {p} fp {fp} fn {fn}: exceptions_bits {rnd(exceptions_dl)}')

    return exceptions_dl

def rs_total_bits(ruleset, possible_conds, pos_df, neg_df, bestsubset_dl=False, ret_bestsubset=False, verbosity=0):
    """ Returns total dl of ruleset.

        bestsubset_dl (optional, <bool>): whether to return estimated minimum possible dl were all rules that increase dl to be removed
        ret_bestsubset (optional): whether to return the best subset that was found. Return format will be (<Ruleset>,dl).
    """
    # When calculating best potential dl, paper is unclear whether you're supposed to reevaluate already-visited rules
    # each time you remove a rule (see note 7), but the vague footnote has an iterative quality to it, so probably not.
    # Weka's source code comments that you are not supposed to, and opines that this is "bizarre."
    # Perhaps not recursing so -- and getting a possibly sub-optimal mdl -- could be viewed charitably a greedy time-saver?
    # (After all, it's not like we're performing an exhaustive search of every possible combination anyways...?)

    if type(ruleset) != Ruleset:
        raise TypeError(f'param ruleset in rs_total_bits should be type Ruleset')
    if ret_bestsubset and not bestsubset_dl:
        raise ValueError(f'bestsubset_dl must be True in order to return bestsubset_dl')

    if not bestsubset_dl:
        theory_bits = rs_theory_bits(ruleset, possible_conds, pos_df, neg_df, verbosity=verbosity)
        data_bits = exceptions_bits(ruleset, pos_df, neg_df, verbosity=verbosity)
        if verbosity>=3: print(f'total ruleset bits | {rnd(theory_bits + data_bits)}')
        return theory_bits + data_bits
    else:
        # Collect the dl of each subset
        subset_dls = []
        theory_dl = 0
        if verbosity>=5: print(f'find best potential dl for {ruleset}:')
        for i, rule in enumerate(ruleset.rules): # Separating theory and exceptions dls in this way is a little longer code, but this way you don't have to recalculate theory each time
            subset = Ruleset(ruleset.rules[:i+1])
            rule_theory_dl = r_theory_bits(rule, possible_conds, verbosity=verbosity)
            theory_dl += rule_theory_dl
            exceptions_dl = exceptions_bits(subset, pos_df, neg_df, verbosity=verbosity)
            subset_dls.append(theory_dl + exceptions_dl)
            if verbosity>=5: print(f'subset 0-{i} | dl: {rnd(subset_dls[i])}')


        # Build up the best Ruleset and calculate the mdl
        mdl_ruleset = Ruleset()
        for i, rule, in enumerate(ruleset.rules):
            if i==0 or subset_dls[i] <= subset_dls[i-1]: # Rule i does not worsen the dl
                mdl_ruleset.add(rule)
        if verbosity>=5:
            print(f'subset dls: {[(i,rnd(dl)) for i,dl in enumerate(subset_dls)]}')
            print(f'best potential ruleset: {mdl_ruleset}')
        mdl = rs_total_bits(mdl_ruleset, possible_conds, pos_df, neg_df, bestsubset_dl=False, verbosity=0) # About to print value below
        if verbosity>=5:
            print(f'best potential dl was {rnd(mdl)}')
            print()
        if not ret_bestsubset:
            return mdl
        else:
            return (mdl_ruleset, mdl)



    """
    def grow_alternative(self, initial_rule, disjs, pos_df, neg_df):
        Grow a new rule, either by adding to the initial rule (revision), or beginning with an empty rule (replacement)

            #initial_rule: rule to begin with or empty rule
            #disjs: ruleset excluding the rule we're looking for an alternative for


        rule0 = copy.deepcopy(initial_rule)
        rule1 = copy.deepcopy(rule0)
        print(f'pos_df {len(pos_df)}')
        print(f'neg_df {len(neg_df)}')
        print(f'initial covers n_negs: {len(rule0.covers(neg_df))}')
        print(f'rule1 is {rule1}')
        while len(rule0.covers(neg_df)) > 0 and rule1 is not None: # Stop refining rule if no negative examples remain
            rule1 = base.best_successor(rule0, pos_df, neg_df, eval_on_ruleset=disjs)
            if rule1 is not None:
                rule0 = rule1
        if rule0.isempty():
            return None
        else:
            return rule0

    """

# Are these supposed to be calculated at the end, or summed with each rule?
#def ruleset_exceptions_bits(ruleset, pos_df, neg_df, verbose=False):
#    rem_pos = copy.deepcopy(pos_df)
#    rem_neg = copy.deepcopy(neg_df)
#    total = 0
#    for rule in ruleset.rules:
#        total += exceptions_bits(rule, rem_pos, rem_neg)
#        rem_pos, rem_neg = base.rm_covered(rule, rem_pos, rem_neg)
#    return total

#def total_bits(rule, pos_df, neg_df, verbose=False):
#    total = rule_bits(rule, pos_df, neg_df) + exceptions_bits(rule, pos_df, neg_df)
#    if verbose: print(f'Rule {rule} total_bits {round(total,2)}')
#    return total


#mdl = ruleset_total_bits(ruleset, pos_df, neg_df, bestsubset_dl=False) # Begin with total Ruleset
#if verbosity>=3: print(f'rs original bits| {mdl}\n')

#i = len(ruleset.rules)-1 # Iterate through rules in reverse order
#while i >= 0:
    # Current subset of rules
#    current_subset = Ruleset(pruned_ruleset.rules[:i])
#    current_dl = ruleset_total_bits(current_subset, possible_conds, pos_df, neg_df, bestsubset_dl=False, verbosity=verbosity)

#    if current_dl > mdl:
#        pruned_ruleset.rules.pop(i)
#        pruned_dl = ruleset_total_bits(pruned_ruleset, pos_df, neg_df, bestsubset_dl=False)


    """
    line 1801 https://searchcode.com/codesearch/view/21712674/
    // Remove the pruning data covered by the following
	    // rules, then simply compute the error rate of the
	    // current rule to prune it.  According to Ripper,
	    // it's equivalent to computing the error of the
	    // whole ruleset -- is it true?
	    pruneData = RuleStats.rmCoveredBySuccessives(pruneData,ruleset, position);
	    replace.prune(pruneData, true);
    """
