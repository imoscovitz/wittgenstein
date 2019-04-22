from .base import Cond, Rule, Ruleset

class CatNap:
    """ Categorical Number Maps:
        Optimized code for speeding up pandas filtering of categorical features.
        .covers methods return pandas indices
    """
    def __init__(self, df, feat_subset=None, cond_subset=None, class_feat=None, pos_class=None):
        if class_feat is None:
            self.conds = self.possible_conds(df) if cond_subset is None else cond_subset
            self.cond_maps = dict([(c, set(c.covers(df).index.tolist())) for c in self.conds])

        else:
            # double-check the following line
            self.conds = self.possible_conds(df.drop(class_feat, axis=1)) if cond_subset is None else [c for c in cond_subset if c.feature != class_feat]
            self.cond_maps = dict([(c, set(c.covers(df.drop(class_feat, axis=1)).index.tolist())) for c in self.conds])
            #pos, neg = pos_neg_split(df, class_feat=class_feat, pos_class=pos_class)
            #self.pos_idx = set(pos.index.tolist())
            #self.neg_idx = set(neg.index.tolist())

        self.num_conds = len(self.conds)
        self.num_idx = len(df)
        self.all = set(df.index.tolist())

    def __str__(self):
        return f'<CatNap object: {self.num_conds} Conds covering {self.num_idx} examples>'
    __repr__ = __str__

    def cond_covers(self, cond, subset={}):
        return self.cond_maps.get(cond) \
            if not subset else self.cond_maps.get(cond).intersection(subset)

    def conj(self, conds, subset={}):
        return set.intersection(*[self.cond_maps.get(c) for c in conds]) \
            if not subset else set.intersection(*[self.cond_maps.get(c) for c in conds]).intersection(subset)

    def rule_covers(self, rule, subset={}): # Same as conj
        # Is there a cleaner way to handle empty rule?
        if rule.conds:
            covered = set.intersection(*[self.cond_maps.get(c) for c in rule.conds])
            return covered if not subset else covered.intersection(subset)
        else:
            return self.all if not subset else self.all.intersection(subset)

    def ruleset_covers(self, ruleset, subset={}):
        return set.union(*[set.intersection(*[self.cond_maps.get(c) for c in r.conds]) for r in ruleset]) \
            if not subset else set.union(*[set.intersection(*[self.cond_maps.get(c) for c in r.conds]) for r in ruleset]).intersection(subset)

    def to_df(self, coverage):
        return df.loc[sorted(list(coverage))]

    def possible_conds(self, df):
        conds = []
        for feat in df.columns.values:
            for val in df[feat].unique():
                conds.append(Cond(feat, val))
        return conds
