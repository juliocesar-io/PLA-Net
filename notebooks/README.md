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

docker run \
    -it --rm --gpus all \
    -v "$(pwd)":/workspace \
    pla-net:latest \
    /bin/bash
