import pandas as pd
import pytest

from wittgenstein.discretize import BinTransformer


def test_bin_ranges_are_flush():
    df = pd.read_csv("credit.csv")
    bin_transformer_ = BinTransformer()
    bin_transformer_.fit(df)
    for feat, bins in bin_transformer_.bins_.items():
        prev_ceil = None
        for bin in bin_transformer_._strs_to_intervals(bins):
            floor, ceil = bin.left, bin.right
            assert prev_ceil is None or floor == prev_ceil
            prev_ceil = ceil


def test_each_bin_in_order():
    df = pd.read_csv("credit.csv")
    bin_transformer_ = BinTransformer()
    bin_transformer_.fit(df)
    for feat, bins in bin_transformer_.bins_.items():
        bins = bin_transformer_._strs_to_intervals(bins)
        assert bins == sorted(bins)


def test_boundless_min_max_bins():
    df = pd.read_csv("credit.csv")
    bin_transformer_ = BinTransformer()
    bin_transformer_.fit(df)
    for feat, bins in bin_transformer_.bins_.items():
        prev_ceil = None
        bins = bin_transformer_._strs_to_intervals(bins)
        assert bins[0].left == float('-inf')
        assert bins[-1].right == float('inf')

def test_fewer_bins_than_n_discretize_bins():
    df = pd.read_csv("credit.csv")
    for n in range(2, 20, 5):
        bin_transformer_ = BinTransformer(n_discretize_bins=n)
        bin_transformer_.fit(df)
        for feat, bin_ranges in bin_transformer_.bins_.items():
            assert len(bin_ranges) <= n


def test_no_bins():
    old_df = pd.read_csv("credit.csv")
    df = old_df.copy()
    bin_transformer_ = BinTransformer(n_discretize_bins=0)
    bin_transformer_.fit(df)
    bin_transformer_.transform(df)
    assert df.equals(old_df)
