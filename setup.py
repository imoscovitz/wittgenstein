import setuptools
#from distutils.core import setup
setuptools.setup(
  name = 'ruleset',
  version = '0.1.0',
  license='MIT',
  description = 'Implementation of ruleset covering algorithms for explainable machine learning',
  author = 'Ilan Moscovitz',
  author_email = 'ilan.moscovitz@gmail.com',
  url = 'https://github.com/imoscovitz/ruleset',
  keywords = ['Classification', 'Decision Rule', 'Machine Learning', 'Explainable Machine Learning'],
  packages=setuptools.find_packages(),
  install_requires=[
          'pandas',
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  include_package_data=True,
  package_data={'': ['data/*.csv']}
)
