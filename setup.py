from setuptools import setup

setup(
  name = 'mklaren',
  packages = ['mklaren', 'mklaren.kernel', 'mklaren.mkl', 'mklaren.projection', 'mklaren.regression',
              'mklaren.util'],
  version = '1.2',
  description = 'The Multiple Kernel Learning Python Library.',
  author = 'Martin Strazar',
  author_email = 'martin.strazar@gmail.com',
  url = 'https://github.com/mstrazar/mklaren',
  download_url = 'https://github.com/mstrazar/mklaren/archive/1.1.tar.gz',
  keywords = ['machine learning', 'kernel methods', 'low-rank approximation', 'regression'],
  classifiers = [],
  test_suite="nose.collector",
  tests_require=['nose'],
  install_requires=["numpy",
                    "scipy>=0.19.0",
                    "scikit-learn>=0.18.1",
                    "cvxopt",
                    ],
)