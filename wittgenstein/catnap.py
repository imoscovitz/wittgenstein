# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

import pandas as pd

from .base import Cond, Rule, Ruleset


class CatNap:
    """Optimized, in-places obnoxiously-dense code, for speeding up search of categorical features.

    "Covers" functions are intended to compute Cond/Rule/Ruleset coverage on dataset indices.
    """

    def __init__(
        self,
        df_or_arr,
        columns=None,
        feat_subset=None,
        cond_subset=None,
        class_feat=None,
        pos_class=None,
    ):
        df = pd.DataFrame(df_or_arr)

        if columns:
            df.columns = columns

        if class_feat is None:
            self.conds = self.possible_conds(df) if cond_subset is None else cond_subset
            self.cond_maps = dict(
                [(c, set(c.covers(df).index.tolist())) for c in self.conds]
            )

        else:
            self.conds = (
                self.possible_conds(df.drop(class_feat, axis=1))
                if cond_subset is None
                else [c for c in cond_subset if c.feature != class_feat]
            )
            self.cond_maps = dict(
                [
                    (c, set(c.covers(df.drop(class_feat, axis=1)).index.tolist()))
                    for c in self.conds
                ]
            )

        self.num_conds = len(self.conds)
        self.num_idx = len(df)
        self.all = set(df.index.tolist())

    def __str__(self):
        return (
            f"<CatNap object: {self.num_conds} Conds covering {self.num_idx} examples>"
        )

    __repr__ = __str__

    def cond_covers(self, cond, subset=None):
        return (
            self.cond_maps.get(cond, set())
            if subset is None
            else self.cond_maps.get(cond, set()).intersection(subset)
        )

    def rule_covers(self, rule, subset=None):
        if rule.conds:
            covered = set.intersection(
                *[self.cond_maps.get(c, set()) for c in rule.conds]
            )
            return covered if subset is None else covered.intersection(subset)
        else:
            return self.all if subset is None else self.all.intersection(subset)

    def ruleset_covers(self, ruleset, subset=None):
        allpos, allneg = ruleset._check_allpos_allneg(warn=False)
        if allpos:
            return self.all if not subset else subset
        elif allneg:
            return set()
        else:
            return (
                set.union(
                    *[
                        set.intersection(
                            *[self.cond_maps.get(c, set()) for c in r.conds]
                        )
                        for r in ruleset
                    ]
                )
                if subset is None
                else set.union(
                    *[
                        set.intersection(
                            *[self.cond_maps.get(c, set()) for c in r.conds]
                        )
                        for r in ruleset
                    ]
                ).intersection(subset)
            )

    def to_df(self, coverage):
        return df.loc[sorted(list(coverage))]

    def possible_conds(self, df):
        conds = []
        for feat in df.columns.values:
            for val in df[feat].unique():
                conds.append(Cond(feat, val))
        return conds

    def pos_idx_neg_idx(
        self, df=None, class_feat=None, pos_class=None, pos_df=None, neg_df=None
    ):
        """Pass in df, pos_class, and class_feat or pos_df and neg_df."""
        if pos_df is None and neg_df is None:
            pos_df = df[df[class_feat] == pos_class]
            neg_df = df[df[class_feat] != pos_class]

        pos_idx = set(pos_df.index.tolist())
        neg_idx = set(neg_df.index.tolist())

        return pos_idx, neg_idx
