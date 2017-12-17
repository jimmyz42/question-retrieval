sudo apt-get update
sudo apt-get install git
git clone https://github.com/jimmyz42/question-retrieval.git
cd question-retrieval
git clone https://github.com/taolei87/askubuntu.git
git clone https://github.com/jiangfeng1124/Android.git
cd askubuntu
gunzip text_tokenized.txt.gz
gunzip vector/vectors_pruned.200.txt.gz
cd ..
cd Android
gunzip corpus.tsv.gz
cd ..
cd ..

curl https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O
chmod +x Miniconda2-latest-Linux-x86_64.sh -O
./Miniconda2-latest-Linux-x86_64.sh

conda install scikit-learn
conda install pytorch torchvision -c pytorch
conda install jupyter
pip install tqdm
pip install prettytable
