conda create --name trinidi python==3.9
conda activate trinidi

pip install -r requirements.txt
pip install -e .


pip install ipython pyqt5
pip install pre-commit autoflake isort black pylint
