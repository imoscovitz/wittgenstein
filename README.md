# Roman!

Since Python was named after the Monte Python comedy troupe, it seems fitting to open the Roman package with a question:

> ... but apart from better sanitation and medicine and education and irrigation and public health and roads and a freshwater system and baths and public order... what have the Romans done for us?
>
>  -- _The Life Of Brian_, Monty Python

The answer: Roman Numerals!

## Purpose

This is a "hello world" example of how to create a Python package. It does so by going through the process of creating a package (`roman`) that
- converting between roman numeral strings and integers
- converting between temperatures

This guide has been broken into several tags, so that the process can be followed. You can access different tag, clone this repo and use
```python
git checkout tags/<tag name>
```

## Tags available

A series of [blog posts](https://kiwidamien.github.io/making-a-python-package.html) walks through the process of creating a package from start to finish. The tags made available are

1. [v0.1: the idea](https://github.com/kiwidamien/Roman/tree/v0.1) 
  We write the Roman numerals function, write a short module, and can import it from one directory.
2. [v0.2: documentation](https://github.com/kiwidamien/Roman/tree/v0.2)
  We write the docstrings of the function, and show best practices.
3. [v0.3: make a package](https://github.com/kiwidamien/Roman/tree/v0.3)  
  We write setup.py, and show how to install on our own system. We can also show how to install from github. 
4. [v0.4: writing tests](https://github.com/kiwidamien/Roman/tree/v0.4)
  We write some unit tests, and introduce the `tox` package to allow easy testing.
5. [v0.5: deploying](https://github.com/kiwidamien/Roman/tree/v0.5)
  We use (twine)[https://pypi.org/project/twine/] to post onto TestPyPI. 

## Installation

To install from tag v0.3 onward, use
```bash
# installation is easy!
$ python setup.py install
```
 
To uninstall (e.g. when updating) you can use `pip`, _even if you are only working locally_
```bash
# so is uninstalling!
$ pip uninstall roman
```

## Usage

Once installed, we can use this module with
```python
>>> import roman
>>> roman.int_to_roman_string(4)
'IV'
>>> roman.roman_string_to_int('MMC')
2100
>>> roman.convert_all(0, 'C')
{'K': 273.15, 'F': 32, 'C': 0}
```

## Useful references

- [This useful article on creating your first Python package](https://medium.com/38th-street-studios/creating-your-first-python-package-181c5e31f3f8)
- [The tox documentation](https://tox.readthedocs.io/en/latest/)
- [The Python code on Project structure](https://docs.python-guide.org/writing/structure/)
- [The Flask MegaTutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world) for helping set up the git tag structure
