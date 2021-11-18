#!/bin/sh

pip install gdown
pip install git+https://github.com/ContinualAI/avalanche.git
pip install scikit-learn==1.0

mkdir data

if [ $1 == "CUB" ]
then
    echo "Setting-up CUB-200-2011 dataset...";
    mv ./CUB_200_2011.tgz ./data/
    tar -xvf ./data/CUB_200_2011.tgz -C ./data
    rm ./data/CUB_200_2011.tgz
    echo "done."

elif [ $1 == "Omniglot" ]
then
    echo "Setting-up Omniglot dataset..."
    mv ./omniglot.zip ./data/
    unzip ./data/omniglot.zip -d ./data/
    mv ./data/content/data/omniglot ./data/
    rm ./data/omniglot.zip
    rm -rf ./data/content
    echo "done."
fi