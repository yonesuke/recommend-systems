pip3 install --upgrade pip
pip3 install --user -r requirements.txt
wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P data
unzip data/ml-10m.zip -d data/