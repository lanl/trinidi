.. _installation:

Installation
============

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


5. Verify that TRINIDI is running correctly, i.e.:

   ::

      python examples/reconstruction_demo.py
