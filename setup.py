from setuptools import setup, find_packages
#from distutils.core import setup
setup(
  name = 'ruleset',
 # packages = ['ruleset'],   # Chose the same as "name"
  version = '0.0.0',
  license='MIT',
  description = 'Implementation of ruleset covering algorithms',   # Give a short description about your library
  author = 'Ilan Moscovitz',
  author_email = 'ilan.moscovitz@gmail.com',
  url = 'https://github.com/imoscovitz/ruleset',   # Provide either the link to your github or to your website
 # download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
 # keywords = ['Classification', 'Decision Rule', 'Machine Learning'],   # Keywords that define your package best
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
  package_data={'': ['data/*.csv']},
)
