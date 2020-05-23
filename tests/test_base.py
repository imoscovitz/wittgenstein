import pytest
import pandas as pd

from wittgenstein.base_functions import (
    pos,
    neg,
    pos_neg_split,
    df_shuffled_split,
    num_pos,
    num_neg,
    df_shuffled_split,
    set_shuffled_split,
    random_split,
)
from wittgenstein.base import Cond, Rule, Ruleset, ascond, asrule, asruleset

FULL_DF = pd.read_csv("house-votes-84.csv")
FIRST_10_EXAMPLES = FULL_DF.head(10)
CLASS_FEAT = "Party"
POS_CLASS = "democrat"
SPLIT_SIZE = 0.6

FEATURE1, VALUE1 = "Handicapped-infants", "n"
FEATURE2, VALUE2 = "Water-project-cost-sharing", "y"
FEATURE3, VALUE3 = "anti-satellite-test-ban", "n"
FEATURE4, VALUE4 = "el-salvador-aid", "y"
COND1 = Cond(FEATURE1, VALUE1)
COND2 = Cond(FEATURE2, VALUE2)
COND3 = Cond(FEATURE3, VALUE3)
COND4 = Cond(FEATURE4, VALUE4)

##### Test Helpers ######


def test_len_pos_df_is_6():
    assert len(pos(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)) == 6


def test_len_neg_df_is_4():
    assert len(neg(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)) == 4


def test_split_pos_is_6_neg_is_4():
    pos_df, neg_df = pos_neg_split(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)
    assert (len(pos_df), len(neg_df)) == (6, 4)


def test_shuffled_splits_are_len_7_len_3():
    df1, df2 = df_shuffled_split(FIRST_10_EXAMPLES, 0.7, random_state=None)
    assert (len(df1), len(df2)) == (7, 3)


def test_num_pos_is_6_num_neg_is_4():
    p = num_pos(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)
    n = num_neg(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)
    assert (p, n) == (6, 4)


def test_num_pos_is_0_num_neg_is_0():
    pos_df, neg_df = pos_neg_split(FIRST_10_EXAMPLES, CLASS_FEAT, POS_CLASS)
    p = num_pos(neg_df, CLASS_FEAT, POS_CLASS)
    n = num_neg(pos_df, CLASS_FEAT, POS_CLASS)
    assert (p, n) == (0, 0)


####### Test Conds ######


def test_conds_are_equal():
    assert Cond("feature", "value") == Cond("feature", "value")


def test_conds_are_unequal_value():
    assert Cond("feature", "value1") != Cond("feature", "value2")


def test_conds_are_unequal_feature():
    assert Cond("feature1", "value") != Cond("feature2", "value")


def test_cond_covers():
    cond = Cond(FEATURE1, VALUE1)
    assert set(cond.covers(FIRST_10_EXAMPLES)[FEATURE1].tolist()) == {VALUE1}


def test_cond_doesnt_cover():
    cond = Cond(FEATURE1, VALUE1)
    assert (
        len(cond.covers(FIRST_10_EXAMPLES[FIRST_10_EXAMPLES[FEATURE1] != VALUE1])) == 0
    )


def test_cond_num_covered_is_7():
    cond = Cond(FEATURE1, VALUE1)
    assert len(cond.covers(FIRST_10_EXAMPLES)[FEATURE1].tolist()) == 7


##### Test Rule #####


def test_empty_rules_equal():
    assert Rule() == Rule()


def test_rules_len_1_equal():
    assert Rule([COND1]) == Rule([COND1])


def test_rules_len_1_unequal():
    assert Rule([COND1]) != Rule([COND2])


def test_rules_len_2_equal():
    rule1 = Rule([COND1, COND2])
    rule2 = Rule([COND1, COND2])
    assert rule1 == rule2


def test_disordered_rules_len_2_equal():
    rule1 = Rule([COND1, COND2])
    rule2 = Rule([COND2, COND1])
    assert rule1 == rule2


def test_rules_len_2_unequal():
    rule1 = Rule([COND1, COND2])
    rule2 = Rule([COND2, COND3])
    assert rule1 != rule2


def test_rule_covers():
    rule = Rule([COND1, COND4])
    rule_covers = rule.covers(FIRST_10_EXAMPLES)
    assertion_df = FIRST_10_EXAMPLES[
        (FIRST_10_EXAMPLES[FEATURE1] == VALUE1)
        & (FIRST_10_EXAMPLES[FEATURE4] == VALUE4)
    ]
    assert set(rule_covers.index.tolist()) == set(assertion_df.index.tolist())


def test_rule_num_covered():
    rule = Rule([COND1, COND4])
    rule_num_covered = rule.num_covered(FIRST_10_EXAMPLES)
    len_assertion_df = len(
        FIRST_10_EXAMPLES[
            (FIRST_10_EXAMPLES[FEATURE1] == VALUE1)
            & (FIRST_10_EXAMPLES[FEATURE4] == VALUE4)
        ]
    )
    assert rule_num_covered == len_assertion_df


def test_ascond():
    assert ascond(Cond("hello", "world")) == Cond("hello", "world")
    assert ascond(("hello", "world")) == Cond("hello", "world")
    assert ascond(("hello=world")) == Cond("hello", "world")


def test_asrule():
    assert asrule(Rule([Cond("hello", "world")])) == Rule([Cond("hello", "world")])
    assert asrule(["hello=world"]) == Rule([Cond("hello", "world")])
    assert asrule("[hello=world]") == Rule([Cond("hello", "world")])


def test_asruleset():
    assert asruleset(Ruleset([Rule([Cond("hello", "world")])])) == Ruleset(
        [Rule([Cond("hello", "world")])]
    )
    assert asruleset("[[hello=world]]") == Ruleset([Rule([Cond("hello", "world")])])
    assert asruleset(["[hello=world]"]) == Ruleset([Rule([Cond("hello", "world")])])


def test_random_split_set_shuffled_split_are_same():
    ssp1, ssp2 = set_shuffled_split(range(len(FULL_DF)), 0.66, random_state=42)
    rs1, rs2 = random_split(range(len(FULL_DF)), 0.66, res_type=list, random_state=42)
    assert set(rs1) == set(ssp1)
    assert set(rs2) == set(ssp2)


def test_random_split_df_shuffled_split_are_same():
    idx1, idx2 = random_split(FULL_DF.index, 0.66, res_type=set, random_state=42)
    df1, df2 = df_shuffled_split(FULL_DF, split_size=0.66, random_state=42)
    assert (set(FULL_DF.loc[idx1, :].index), set(FULL_DF.loc[idx2, :].index)) == (
        idx1,
        idx2,
    )
