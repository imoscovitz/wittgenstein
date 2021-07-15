from copy import deepcopy
import os

import pytest
import pandas as pd

from wittgenstein.irep import IREP
from wittgenstein.ripper import RIPPER
from wittgenstein.base import Ruleset, ruleset_fromstr, rule_fromstr

DF = pd.read_csv("mushroom.csv")
class_feat = "Poisonous/Edible"
pos_class = "p"
original_ruleset_str = "[[Odor=f] V [Gill-size=n] V [Spore-print-color=r] V [Odor=m]]"
original_ruleset = ruleset_fromstr(original_ruleset_str)
original_rules = original_ruleset.rules
original_irep = IREP(random_state=42)
original_irep.fit(DF, class_feat="Poisonous/Edible", pos_class="p")
# Ensure setup works
assert original_ruleset == original_irep.ruleset_


credit_df = pd.read_csv("credit.csv")
credit_class_feat = "Class"
credit_pos_class = "+"
credit_rip = RIPPER(random_state=42, verbosity=0)
credit_rip.fit(credit_df, class_feat="Class", pos_class="+")
credit_original_ruleset = ruleset_fromstr(
    "[[A9=t^A10=t^A14=0] V [A9=t^A10=t^A15=1000-4607] V [A9=t^A10=t^A11=3-7^A12=f] V [A9=t^A10=t] V [A9=t^A7=h^A6=q] V [A9=t^A14=0^A4=u] V [A9=t^A15=4607-100000]]"
)
assert credit_rip.ruleset_ == credit_original_ruleset


def test_initruleset():
    # IREP
    empty_ruleset = Ruleset()
    irep = IREP(random_state=42)
    irep.init_ruleset(empty_ruleset, class_feat, pos_class)
    assert irep.ruleset_ == empty_ruleset
    assert irep.class_feat == class_feat
    assert irep.pos_class == pos_class

    irep = IREP(random_state=42)
    irep.init_ruleset(original_ruleset, class_feat, pos_class)
    assert irep.ruleset_ == original_ruleset
    assert irep.class_feat == class_feat
    assert irep.pos_class == pos_class

    irep = IREP(random_state=42)
    irep.init_ruleset(original_ruleset_str, class_feat, pos_class)
    assert irep.ruleset_ == original_ruleset
    assert irep.class_feat == class_feat
    assert irep.pos_class == pos_class

    # RIPPER
    empty_ruleset = Ruleset()
    rip = RIPPER(random_state=42)
    rip.init_ruleset(empty_ruleset, class_feat, pos_class)
    assert rip.ruleset_ == empty_ruleset
    assert rip.class_feat == class_feat
    assert rip.pos_class == pos_class

    rip = RIPPER(random_state=42)
    rip.init_ruleset(original_ruleset, class_feat, pos_class)
    assert rip.ruleset_ == original_ruleset
    assert rip.class_feat == class_feat
    assert rip.pos_class == pos_class

    rip = RIPPER(random_state=42)
    rip.init_ruleset(original_ruleset_str, class_feat, pos_class)
    assert rip.ruleset_ == original_ruleset
    assert rip.class_feat == class_feat
    assert rip.pos_class == pos_class


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
    assert (
        irep.ruleset_.rules
        == original_rules[:2] + [rule_fromstr(new_rule)] + original_rules[2:]
    )

    # By value
    irep = deepcopy(original_irep)
    irep.insert_rule("[Odor=f]", new_rule)
    assert irep.ruleset_.rules == [rule_fromstr(new_rule)] + original_rules

    irep = deepcopy(original_irep)
    irep.insert_rule("[Spore-print-color=r]", new_rule)
    assert (
        irep.ruleset_.rules
        == original_rules[:2] + [rule_fromstr(new_rule)] + original_rules[2:]
    )


def test_replace_rule():
    # By index
    new_rule = "[Gill-size=y]"

    irep = deepcopy(original_irep)
    irep.replace_rule_at(1, new_rule)
    assert (
        irep.ruleset_.rules
        == [original_rules[0]] + [rule_fromstr(new_rule)] + original_rules[2:]
    )

    # By value
    irep = deepcopy(original_irep)
    irep.replace_rule("[Gill-size=n]", new_rule)
    assert (
        irep.ruleset_.rules
        == [original_rules[0]] + [rule_fromstr(new_rule)] + original_rules[2:]
    )


def test_save_load_csv():
    csv_filename = "temp_test_sme.py_test_save_load.csv"
    rip = credit_rip.copy()
    rip.remove_rule_at(1)
    rip.remove_rule_at(1)
    rip.add_rule("[A8=-2.5--1.2]")
    # Make sure set up works:
    assert rip.ruleset_ == ruleset_fromstr(
        "[[A9=t^A10=t^A14=0] V [A9=t^A10=t] V [A9=t^A7=h^A6=q] V [A9=t^A14=0^A4=u] V [A9=t^A15=4607-100000] V [A8=-2.5--1.2]]"
    )
    # Save
    rip.to_csv(csv_filename)
    loaded_rip = RIPPER(random_state=42)
    # Load
    loaded_rip.from_csv(
        csv_filename, class_feat=credit_class_feat, pos_class=credit_pos_class
    )
    os.remove(csv_filename)
    assert loaded_rip.ruleset_ == rip.ruleset_
    assert loaded_rip.bin_transformer_.bins_ == rip.bin_transformer_.bins_
    assert loaded_rip.bin_transformer_.n_discretize_bins == 10
    assert loaded_rip.bin_transformer_.names_precision == 1
    assert loaded_rip.bin_transformer_.verbosity == 0
    assert loaded_rip.class_feat == credit_class_feat
    assert loaded_rip.pos_class == credit_pos_class


def test_save_load_txt():
    txt_filename = "temp_test_sme.py_test_save_load.txt"
    rip = credit_rip.copy()
    rip.remove_rule_at(1)
    rip.remove_rule_at(1)
    rip.add_rule("[A8=-2.5--1.2]")
    # Make sure set up works:
    assert rip.ruleset_ == ruleset_fromstr(
        "[[A9=t^A10=t^A14=0] V [A9=t^A10=t] V [A9=t^A7=h^A6=q] V [A9=t^A14=0^A4=u] V [A9=t^A15=4607-100000] V [A8=-2.5--1.2]]"
    )
    # Save
    rip.to_txt(txt_filename)
    loaded_rip = RIPPER(random_state=42)
    # Load
    loaded_rip.from_txt(
        txt_filename, class_feat=credit_class_feat, pos_class=credit_pos_class
    )
    os.remove(txt_filename)
    assert loaded_rip.ruleset_ == rip.ruleset_
    assert loaded_rip.bin_transformer_.bins_ == {
        "A11": [("7", "16")],
        "A8": [("-2.5", "-1.2")],
    }
    assert loaded_rip.bin_transformer_.n_discretize_bins == 10
    assert loaded_rip.bin_transformer_.names_precision == 1
    assert loaded_rip.bin_transformer_.verbosity == 0
    assert loaded_rip.class_feat == credit_class_feat
    assert loaded_rip.pos_class == credit_pos_class


def test_use_initial_model():

    initial_model = "[[A9=t^A6=w] ^ [hello=world]]"
    expected_irep = ruleset_fromstr(
        """
        [[A9=t^A6=w^hello=world] V
        [A9=t^A10=t] V
        [A9=t^A7=h] V
        [A9=t^A4=u^A7=v]]
        """
        #[[A9=t^A10=t^hello=world] V [A9=t^A10=t] V [A9=t^A7=h] V [A9=t^A4=u^A7=v]]
    )
    expected_rip = ruleset_fromstr(
        """
        [[A9=t^A6=w^hello=world] V
        [A9=t^A10=t] V
        [A9=t^A7=h] V
        [A9=t^A4=u^A14=0^A15=0-0] V
        [A9=t^A6=w]]
        """
    )

    updated_credit_df = credit_df.copy()
    updated_credit_df['hello'] = 'earth2'

    # From str
    irep = IREP(random_state=1)
    irep.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_model
    )
    assert irep.ruleset_ == expected_irep
    rip = RIPPER(random_state=1)
    rip.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_model
    )
    assert rip.ruleset_ == expected_rip

    # From IREP to IREP/RIPPER
    initial_irep_model = IREP(random_state=1)
    initial_irep_model.init_ruleset(
        initial_model,
        class_feat='Poisonous/Edible',
        pos_class='p'
    )
    print('initial irep model')
    print(initial_irep_model.ruleset_)
    irep = IREP(random_state=1, verbosity=1)
    irep.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_irep_model
    )
    print("OUTPUT`")
    print(irep.ruleset_)
    print()
    print(expected_irep)
    assert irep.ruleset_ == expected_irep
    rip = RIPPER(random_state=1)
    rip.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_irep_model
    )
    assert rip.ruleset_ == expected_rip

    # From RIP
    initial_rip_model = RIPPER()
    initial_rip_model.init_ruleset(initial_model, class_feat='Poisonous/Edible', pos_class='p')
    irep = IREP(random_state=1)
    irep.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_rip_model
    )
    assert irep.ruleset_ == expected_irep
    rip = RIPPER(random_state=1)
    rip.fit(updated_credit_df, class_feat='Class', pos_class='+',
            initial_model=initial_rip_model
    )
    assert rip.ruleset_ == expected_rip

    # No side-effects
    assert initial_irep_model.ruleset_ == ruleset_fromstr(initial_model)
    assert initial_rip_model.ruleset_ == ruleset_fromstr(initial_model)
