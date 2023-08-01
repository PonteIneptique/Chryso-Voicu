#!/bin/sh

virtualenv -p python3.8 bert-env
#virtualenv -p python3.8 papie-env

# Install Greek Bert
bert-env/bin/pip install transformers==4.19.4 flair==0.11.3 pandas
# If you want to install for a specific CUDA
bert-env/bin/pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

# Download BERT POS Model
mkdir ./LM/
# Might be required ?
git lfs install
GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/pranaydeeps/Ancient-Greek-BERT ./LM/SuperPeitho-v1

wget https://github.com/pranaydeeps/Ancient-Greek-BERT/raw/main/SuperPeitho-FLAIR-v2/final-model.pt SuperPeitho-FLAIR-v2/final-model.pt

# Install Pie-Extended
#papie-env/bin/pip install pie-extended==0.0.40 pandas==1=23=1
#papie-env/bin/pip install torch==1.8.1 --extra-index-url https://download.pytorch.org/whl/cu111 --upgrade
#papie-env/bin/pie-extended download grc
