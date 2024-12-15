#!/bin/bash

# run this as `$source setup_venv.sh`

# 1. Create a Python Virtual Environment named 'nlp-venv'
if [ ! -d "nlp-venv" ]; then
  python3 -m venv nlp-venv # Mac/Linux
fi
# or on Windows:
# python -m venv nlp-venv

# 2. Activate the Virtual Environment
source nlp-venv/bin/activate  # Mac/Linux
# On Windows:
# nlp-venv\Scripts\activate

# 3. Install Required Python Packages
pip install --upgrade pip
pip install pandas openai requests tqdm tenacity datasets scikit-learn

echo "Virtual environment setup complete and dependencies installed."

# run `deactivate` to exit the venv