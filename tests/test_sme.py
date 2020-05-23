import pytest
import pandas as pd

from wittgenstein.irep import IREP
from wittgenstein.base import Ruleset, ruleset_fromstr, rule_fromstr

DF = pd.read_csv("mushroom.csv")
original_ruleset_str = "[[Odor=f] V [Gill-size=n] V [Spore-print-color=r] V [Odor=m]]"
original_ruleset = ruleset_fromstr(original_ruleset_str)
original_rules = original_ruleset.rules


def test_initruleset():
    irep = IREP(random_state=42)
    irep.init_ruleset()
    irep.ruleset_ == Ruleset()

    irep = IREP(random_state=42)
    irep.init_ruleset(original_ruleset)
    irep.ruleset_ == original_ruleset

    irep = IREP(random_state=42)
    irep.init_ruleset(original_ruleset_str)
    irep.ruleset_ == original_ruleset


def test_add_rule():
    irep = IREP(random_state=42)
    irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
    new_rule = "[Spore-print-color=r^Stalk-surface-above-ring=k]"
    irep.add_rule(new_rule)
    assert irep.ruleset_.rules == original_rules + [irep.ruleset_.rules[-1]]


def test_remove_rule():
    irep = IREP(random_state=42)
    irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
    irep.remove_rule(1)
    assert irep.ruleset_.rules == [original_rules[0]] + original_rules[2:]


def test_insert_rule():
    new_rule = rule_fromstr("[hello=world]")
    irep = IREP(random_state=42)
    irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
    irep.insert_rule(0, new_rule)
    assert irep.ruleset_.rules == [new_rule] + original_rules

    irep = IREP(random_state=42)
    irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
    irep.insert_rule(2, new_rule)
    assert irep.ruleset_.rules == original_rules[:2] + [new_rule] + original_rules[2:]


def test_edit_rule():
    new_rule = rule_fromstr("[Gill-size=y]")
    irep = IREP(random_state=42)
    irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
    irep.edit_rule(1, new_rule)
    assert irep.ruleset_.rules == [original_rules[0]] + [new_rule] + original_rules[2:]
