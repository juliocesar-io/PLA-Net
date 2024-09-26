## Using a Notebook locally



## Using Docker 

build the docker image, and make sure you have NVIDIA GPU and NVIDIA Container Toolkit installed.

docker build -t pla-net:latest .


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

# Gradio Server

docker run \
    -it --rm --gpus all \
    -p 7860:7860 \
    -v "$(pwd)":/workspace \
    pla-net:latest \
    python app.py