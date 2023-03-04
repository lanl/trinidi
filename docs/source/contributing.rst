.. _contributing:

Contributing
============

.. raw:: html

    <style type='text/css'>
    div.document ul blockquote {
       margin-bottom: 8px !important;
    }
    div.document li > p {
       margin-bottom: 4px !important;
    }
    div.document ul > li {
      list-style: square outside !important;
      margin-left: 1em !important;
    }
    section {
      padding-bottom: 1em;
    }
    ul {
      margin-bottom: 1em;
    }
    </style>


Contributions to TRINIDI are welcome.


Installing a Development Version
--------------------------------

1. Fork the ``trinidi`` repository, creating a copy in your own git account.

2. Make sure that you have Python 3.7 or later installed in order to create a conda virtual environment.

3. Clone your fork from the source repo.

   ::

      git clone git@github.com:<username>/trinidi.git


4. Create a conda environment using Python 3.7 or later, e.g.:

   ::

      conda create -n trinidi python=3.9


5. Activate the created conda virtual environment:

   ::

      conda activate trinidi


6. Change directory to the root of the cloned repository:

   ::

      cd trinidi


7. Add the TRINIDI repo as an upstream remote to sync your changes:

   ::

      git remote add upstream https://www.github.com/lanl/trinidi


8. After adding the upstream, the recommended way to install TRINIDI and its dependencies is via pip:

   ::

      pip install -r requirements.txt  # Installs basic requirements
      pip install -r dev_requirements.txt  # Installs developer requirements
      pip install -r docs/docs_requirements.txt # Installs documentation requirements
      pip install -e .  # Installs TRINIDI from the current directory in editable mode


9. The TRINIDI project uses the `black <https://black.readthedocs.io/en/stable/>`_,
   `isort <https://pypi.org/project/isort/>`_ and `pylint <https://pylint.pycqa.org/en/latest/>`_
   code formatting utilities. It is important to set up a `pre-commit hook <https://pre-commit.com>`_ to
   ensure that any modified code passes format check before it is committed to the development repo:

   ::

      pre-commit install  # Sets up git pre-commit hooks

   It is also recommended to `pin the conda package version
   <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning>`__
   of `black <https://black.readthedocs.io/en/stable/>`_ to the version
   number specified in ``dev_requirements.txt``.





Tests
-----

All functions and classes should have corresponding ``pytest`` unit tests.


Running Tests
^^^^^^^^^^^^^


To be able to run the tests, install ``pytest`` and, optionally,
``pytest-runner``:

::

    conda install pytest pytest-runner

The tests can be run by

::

    pytest

or (if ``pytest-runner`` is installed)

::

    python setup.py test

from the TRINIDI repository root directory. Tests can be run in an installed
version of TRINIDI by

::

   pytest --pyargs trinidi
