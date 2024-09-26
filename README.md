# PLA-Net: Predicting Protein-Ligand Interactions with Graph Convolutional Networks for Interpretable Pharmaceutical Discovery

Paola Ruiz Puentes, Laura Rueda-Gensini, Natalia Valderrama, Isabela Hernández, Cristina González, Laura Daza, Carolina Muñoz-Camargo, Juan C. Cruz, Pablo Arbeláez

This repository contains the official implementation of PLA-Net: [Predicting target–ligand interactions with graph convolutional networks for interpretable pharmaceutical discovery](https://www.nature.com/articles/s41598-022-12180-x). 

## Paper

[Predicting target–ligand interactions with graph convolutional networks for interpretable pharmaceutical discovery](https://www.nature.com/articles/s41598-022-12180-x),<br/>
[Paola Ruiz Puentes](https://paolaruizp.github.io)<sup>1,2</sup>, Laura Rueda-Gensini<sup>1,2</sup>, [Natalia Valderrama](https://nfvalderrama.github.io)<sup>1,2</sup>, Isabela Hernández<sup>1,2</sup>, Cristina González<sup>1,2</sup>, [Laura Daza](https://lauradaza.github.io/Laura_Daza/)<sup>1,2</sup>, Carolina Muñoz-Camargo<sup>2</sup>, Juan C. Cruz<sup>2</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup><br/>
Scientific Reports, 2022.<br><br>

<sup>1 </sup> Center  for  Research  and  Formation  in  Artificial  Intelligence .([CINFONIA](https://cinfonia.uniandes.edu.co/)),  Universidad  de  los  Andes,  Bogotá 111711, Colombia. <br/>
<sup>2 </sup> Department  of  Biomedical  Engineering,  Universidad  de  los  Andes,  Bogotá 111711, Colombia.<br/>

## Docker Install

To prevent conflicts with the host machine, it is recommended to run PLA-Net in a Docker container.

First make sure you have an NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed. Then build the image with the following command:

```bash
docker build -t pla-net:latest .
```

### Inference

To run inference, run the following command:

This will run inference for the target protein `ada` with the SMILES in the `input_smiles.csv` file and save the predictions to the `output_predictions.csv` file.

```bash
docker run \
    -it --rm --gpus all \
    -v "$(pwd)":/workspace \
    pla-net:latest \
    python notebooks/scripts/pla_net_inference.py \
    --use_gpu \
    --target ada \
    --target_list /workspace/data/datasets/AD/Targets_Fasta.csv \
    --target_checkpoint_path /workspace/pretrained-models/BINARY_ada \
    --input_file_smiles /workspace/notebooks/example/input_smiles.csv \
    --output_file /workspace/notebooks/example/output_predictions.csv
```

Args:

- `use_gpu`: Use GPU for inference.
- `target`: Target protein ID from the list of targets. Check the list of available targets in the [data](https://github.com/juliocesar-io/PLA-Net/blob/main/data/datasets/AD/Targets_Fasta.csv) folder.
- `target_list`: Path to the target list CSV file.
- `target_checkpoint_path`: Path to the target checkpoint. (e.g. `/workspace/pretrained-models/BINARY_ada`) one checkpoint for each target.
- `input_file_smiles`: Path to the input SMILES file.
- `output_file`: Path to the output predictions file.

## Local Install
The following steps are required in order to run PLA-Net in a local environment:<br />

```bash
$ export PATH=/usr/local/cuda-11.0/bin:$PATH <br />
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH <br />

$ conda create --name PLA-Net <br />
$ conda activate PLA-Net <br />

$ bash env.sh
```

## Models
We provide trained models available for download in the following [link](http://157.253.243.19/PLA-Net/).

## Usage
To train each of the components of our method: LM, LM+Advs, LMPM and PLA-Net please refer to planet.sh file and run the desired models.

To evaluate each of the components of our method: LM, LM+Advs, LMPM and PLA-Net please run the corresponding bash file in the inference folder.

## Citation

We hope you find our paper useful. To cite us, please use the following BibTeX entry:

```
@article{ruiz2022predicting,
  title={Predicting target--ligand interactions with graph convolutional networks for interpretable pharmaceutical discovery},
  author={Ruiz Puentes, Paola and Rueda-Gensini, Laura and Valderrama, Natalia and Hern{\'a}ndez, Isabela and Gonz{\'a}lez, Cristina and Daza, Laura and Mu{\~n}oz-Camargo, Carolina and Cruz, Juan C and Arbel{\'a}ez, Pablo},
  journal={Scientific reports},
  volume={12},
  number={1},
  pages={1--17},
  year={2022},
  publisher={Nature Publishing Group}
}
```
