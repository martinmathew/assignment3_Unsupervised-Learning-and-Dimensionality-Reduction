git clone https://github.com/martinmathew/assignment3_Unsupervised-Learning-and-Dimensionality-Reduction.git

# Change Directory
cd assignment3_Unsupervised-Learning-and-Dimensionality-Reduction

# create env
conda env update --prefix ./env --file environment.yml  --prune

# activate env
conda activate path to env\env

# For Charts related to Credit Card Fraud Detection
python credit_card.py


# For Charts related to Heart Rate Failure Detection
python heart_attack.py
