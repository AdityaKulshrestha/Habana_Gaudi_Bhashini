#!/bin/bash

sudo apt-get update
sudo apt-get install libsndfile1-dev ffmpeg unzip -y

pip install git+https://github.com/huggingface/optimum-habana.git

git clone https://github.com/gokulkarthik/Trainer 
cd Trainer && pip install -e .[all] && cd ..

cd ..

echo "Done TTS, starting Indic-TTS: $(pwd)"

pip install -r Indic_TTS/requirements.txt
pip install numpy==1.23
pip install -r Indic_TTS/inference/requirements-ml.txt 
pip install -r Indic_TTS/inference/requirements-utils.txt
pip install -r Indic_TTS/inference/requirements-server.txt

mkdir -p checkpoints/
mkdir -p models/v1/

wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/bn.zip &> /dev/null
unzip -qq bn.zip
cp -r bn checkpoints/
cp -r bn models/v1/
rm -r bn 

#! TODO: Need to find a more permanent solution
#sed -i '122i\            return' /usr/local/lib/python3.10/dist-packages/fairseq/modules/transformer_layer.py  

pip install ai4bharat-transliteration asteroid
pip uninstall fairseq 
git clone https://github.com/HabanaAI/Fairseq.git
cd Fairseq && pip install --editable ./
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..

# pip install --upgrade pyworld