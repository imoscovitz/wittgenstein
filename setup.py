import setuptools
setuptools.setup(
  name = 'ruleset',
  version = '1.0.0',
  license='MIT',
  description = 'Implementation of ruleset covering algorithms for explainable machine learning',
  long_description=open('DESCRIPTION.rst').read(),
  author = 'Ilan Moscovitz',
  author_email = 'ilan.moscovitz@gmail.com',
  url = 'https://github.com/imoscovitz/ruleset',
  keywords = ['Classification', 'Decision Rule', 'Machine Learning', 'Explainable Machine Learning', 'Data Science'],
  packages=setuptools.find_packages(),
  install_requires=[
          'pandas',
          'numpy'
      ],
  classifiers=[
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  include_package_data=True,
  package_data={'': ['data/*.csv']}
)
