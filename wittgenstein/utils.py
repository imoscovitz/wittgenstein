def _drop_chars(str_, chars):
    res = str_
    for char in chars:
        res = res.replace(char, "")
    return res
