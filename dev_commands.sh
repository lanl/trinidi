# useful
pip install ipython pyqt5

# building docs
pip install pre-commit autoflake isort black pylint
pip install jupyter py2jn
pip install -r docs/docs_requirements.txt
conda install sphinx
cd docs && make clean && make html && open -a "Safari" ../build/sphinx/html/index.html && cd ..



pip install -r docs/docs_requirements.txt # Installs documentation requirements
pre-commit install  # Sets up git pre-commit hooks
