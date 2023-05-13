# RLib

The library that I built while learning some Reinforcement Learning algorithms.

## Installation

### Code

To install the package simply run the following command:

```bash
pip install .
```

Remember that, if using a virtual environment, you must activate it before running the command. Furthermore,
if using ``conda``, you must call ``conda install pip`` before installing the package. Else, the package will
be installed using another ``pip`` that is not the one from the ``conda`` environment.

### Documentation

To generate the documentation, run the following command:

```bash
cd docs
make html
```

The documentation will be generated in the ``docs/build/html`` folder. To open it, simply open the ``index.html`` in your browser.
