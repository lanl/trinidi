.. _installation:

Installation
============

PyPi
----

To install ``TRINIDI`` using the Python Package Index run the command:

::

      pip install trinidi

From Source
-----------

Installing from source downloads the most recent version of `TRINIDI`
and gives you direct access to the
:ref:`Jupyter notebooks <examples_notebooks>` the and corresponding
:ref:`Python scripts <examples_scripts>`.

To install,

1. Clone TRINIDI from the source repo:

   ::

      git clone --recurse-submodules https://github.com/lanl/trinidi.git


2. Make sure that you have Python 3.7 or later installed in order to
   create a conda virtual environment:

   ::

      conda create -n trinidi python=3.9


3. Activate the conda virtual environment:

   ::

      conda activate trinidi



4. After entering the directory, the recommended way to install TRINIDI
   and its dependencies is via pip:

   ::

      cd trinidi
      pip install -r requirements.txt  # Installs basic requirements
      pip install -e .  # Installs TRINIDI in editable mode


5. Verify that TRINIDI is running correctly, e.g. run an example script:

   ::

      python examples/reconstruction_demo.py

The corresponding Jupyter notebooks can be found at
``data/examples/notebooks/``.
