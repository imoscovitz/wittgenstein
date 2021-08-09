# Author: Ilan Moscovitz <ilan.moscovitz@gmail.com>
# License: MIT

from copy import deepcopy
import numpy as np


def drop_chars(str_, chars):
    res = str_
    for char in chars:
        res = res.replace(char, "")
    return res


def remove_duplicates(list_):
    res = deepcopy(list_)
    encountered = set()
    i = 0
    while i < len(res):
        if res[i] in encountered:
            del res[i]
        else:
            encountered.add(res[i])
            i += 1
    return res


def aslist(data):
    try:
        return data.aslist()
    except:
        return data


def try_np_tonum(value):
    try:
        return value.item()
    except:
        return value


def flagged_return(flags, objects):
    """Return only objects with corresponding True flags. Useful for functions with multiple possible return items."""
    if sum(flags) == 1:
        return objects[0]
    elif sum(flags) > 1:
        return tuple([object for flag, object in zip(flags, objects) if flag])
    else:
        return ()


def rnd(float, places=None):
    """Round a float to decimal places.

    float : float
        Value to round.
    places : int, default=None
        Number of decimal places to round to. None defaults to 1 decimal place if float < 100, otherwise defaults to 0 places.
    """
    if places is None:
        if float < 1:
            places = 2
        elif float < 100:
            places = 1
        else:
            places = 0
    rounded = round(float, places)
    if rounded != int(rounded):
        return rounded
    else:
        return int(rounded)


def weighted_avg_freqs(counts):
    """Return weighted mean proportions of counts in the list.

    counts <list<tuple>>
    """
    arr = np.array(counts)
    total = arr.flatten().sum()
    return arr.sum(axis=0) / total if total else arr.sum(axis=0)
