### Mklaren

A Multiple kernel learning Python library.


#### Features
* Support for standard kernel functions (RBF, linear, polynomial, sigmoid)
* Efficient interface to the kernel matrix
* Low-rank kernel approximation methods (Incomplete Cholesky Decomposition, Cholesky with Side-information, the Nystrom method)
* Multiple kernel learning methods based on centered alignment
* Simultaneous multiple kernel learning and low-rank approximation base on least-angle regression (the Mklaren algorithm)


#### Resources

* Quick Start Guide [[jupyter notebook]](https://cdn.rawgit.com/mstrazar/mklaren/master/docs/quick_start.ipynb) [[html]](https://cdn.rawgit.com/mstrazar/mklaren/master/docs/quick_start.html)

* Poster presentation [[png]](https://cdn.rawgit.com/mstrazar/mklaren/master/docs/poster.png)[[eps]](https://cdn.rawgit.com/mstrazar/mklaren/master/docs/poster.eps)

* Mklaren method article [[arXiv]](http://arxiv.org/abs/1601.04366) [[.py scripts]](https://github.com/mstrazar/mklaren/wiki/Experiments-in-the-Mklaren-article)

* Full documentation [[html]](https://github.com/mstrazar/mklaren/tree/master/docs/build/html)

#### Installation


The Mklaren package is heavily based on NumPy and SciPy packages. Make sure these are installed and visible in the
Python environment.

    pip install numpy
    pip install scipy

Mklaren and its dependencies are installed from the PyPI package repository:

    pip install mklaren

Alternatively, the package can be installed by cloning this repository and running:

    python setup.py install

Unit tests are run with:

    python setup.py test


#### Additional dependencies

Certain experiments in the article use additional functionalities, not required strictly by the library.

Some code in the examples uses Matplotlib. It shall be installed manually due to possible system dependencies.

    pip install matplotlib

Running the method CSI (Cholesky with Side Information) assumes a local `octave`
installation as well as Oct2Py python module.

Octave can be installed for your OS from the [Octave website](https://www.gnu.org/software/octave/).
The Python interface to octave is installed separately.

    pip install oct2py

The FITC method is borrowed from the GPy package:

    pip install GPy


#### Building documentation

Run Sphinx inside `docs`

    sphinx-build -b html source/ build/html/
