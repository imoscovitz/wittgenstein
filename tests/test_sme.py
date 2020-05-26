from copy import deepcopy

import pytest
import pandas as pd

from wittgenstein.irep import IREP
from wittgenstein.base import Ruleset, ruleset_fromstr, rule_fromstr

DF = pd.read_csv("mushroom.csv")
original_ruleset_str = "[[Odor=f] V [Gill-size=n] V [Spore-print-color=r] V [Odor=m]]"
original_ruleset = ruleset_fromstr(original_ruleset_str)
original_rules = original_ruleset.rules
original_irep = IREP(random_state=42)
original_irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
# Ensure setup works
assert original_ruleset == original_irep.ruleset_

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


def set_ruleset():
    new_ruleset = ruleset_fromstr("[[Gill-size=y] v [hello=world]")
    irep = deepcopy(original_irep)
    irep.set_ruleset(new_ruleset)
    assert irep.ruleset_ == new_ruleset


def test_add_rule():
    new_rule = "[Spore-print-color=r^Stalk-surface-above-ring=k]"
    irep = deepcopy(original_irep)
    irep.add_rule(new_rule)
    assert irep.ruleset_.rules == original_rules + [irep.ruleset_.rules[-1]]


def test_remove_rule():
    # By index
    irep = deepcopy(original_irep)
    irep.remove_rule_at(1)
    assert irep.ruleset_.rules == [original_rules[0]] + original_rules[2:]

    # By value
    irep = deepcopy(original_irep)
    irep.remove_rule("[Gill-size=n]")
    assert irep.ruleset_.rules == [original_rules[0]] + original_rules[2:]


def test_insert_rule():
    # By index
    new_rule = "[hello=world]"
    irep = deepcopy(original_irep)
    irep.insert_rule_at(0, new_rule)
    assert irep.ruleset_.rules == [rule_fromstr(new_rule)] + original_rules

    irep = deepcopy(original_irep)
    irep.insert_rule_at(2, new_rule)
    assert irep.ruleset_.rules == original_rules[:2] + [rule_fromstr(new_rule)] + original_rules[2:]

    # By value
    irep = deepcopy(original_irep)
    irep.insert_rule("[Odor=f]", new_rule)
    assert irep.ruleset_.rules == [rule_fromstr(new_rule)] + original_rules

    irep = deepcopy(original_irep)
    irep.insert_rule("[Spore-print-color=r]", new_rule)
    assert irep.ruleset_.rules == original_rules[:2] + [rule_fromstr(new_rule)] + original_rules[2:]


def test_replace_rule():
    # By index
    new_rule = "[Gill-size=y]"

    irep = deepcopy(original_irep)
    irep.replace_rule_at(1, new_rule)
    assert irep.ruleset_.rules == [original_rules[0]] + [rule_fromstr(new_rule)] + original_rules[2:]

    # By value
    irep = deepcopy(original_irep)
    irep.replace_rule("[Gill-size=n]", new_rule)
    assert irep.ruleset_.rules == [original_rules[0]] + [rule_fromstr(new_rule)] + original_rules[2:]
