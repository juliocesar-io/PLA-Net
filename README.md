# PLA-Net: Modeling Protein-Ligand Interactions with Graph Convolutional Networks for Interpretable Pharmaceutical Discovery

Paola Ruiz-Puentes, Laura Rueda-Gensini, Natalia Valderrama, Isabela Hernández, Cristina González, Laura Daza, Carolina Muñoz-Camargo, Juan C. Cruz, Pablo Arbeláez

This repository contains the official implementation of PLA-Net, submitted for revision to *Scientific Reports*. 

## Installation
The following steps are required in order to run PLA-Net:

1.
export PATH=/usr/local/cuda-11.0/bin:$PATH **Enter**
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH **Enter**

2.
conda create --name PLA-Net **Enter**
conda activate PLA-Net **Enter**

3.
Run env.sh

## Models
We provide trained models available for download in the following [link](http://157.253.243.19/PLA-Net/).

## Usage
To evaluate each of the components of our method: LM, LM+Advs, LMPM and PLA-Net please run the corresponding bash file.
