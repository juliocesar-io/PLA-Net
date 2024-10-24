import uuid
import gradio as gr
import torch
import os
import pandas as pd
from rdkit import Chem
from scripts.pla_net_inference import main
from utils.args import ArgsInit

os.system("nvidia-smi")
print("TORCH_CUDA", torch.cuda.is_available())

PROJECT_URL = "https://www.nature.com/articles/s41598-022-12180-x"

DEFAULT_PATH_DOCKER = "/home/user/app"

def load_and_filter_data(protein_id, ligand_smiles):
    
    # generate random short id, make short
    random_id = str(uuid.uuid4())[:8]
    
    print("Inference ID: ", random_id)
    
    # check that ligand_smiles is not empty
    if not ligand_smiles or ligand_smiles.strip() == "":
        error_msg = f"!SMILES string is required"
        raise gr.Error(error_msg, duration=5)
    
    # Split the input SMILES string by ':' to get a list
    smiles_list = ligand_smiles.split(':')
    
    
    
    print("Smiles to predict: ", smiles_list)
    print("Target Protein ID: ", protein_id)
    
    # Validate SMILES
    invalid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            invalid_smiles.append(smiles.strip())
            

    
    if invalid_smiles:
        error_msg = f"!Invalid 💥 SMILES string(s) : {', '.join(invalid_smiles)}"
        raise gr.Error(error_msg, duration=5)
    
        # Create tmp folder
    os.makedirs(f"{DEFAULT_PATH_DOCKER}/example/tmp", exist_ok=True)
    
    # Save SMILES to CSV
    df = pd.DataFrame({"smiles": [s.strip() for s in smiles_list if s.strip()]})
    df.to_csv(f"{DEFAULT_PATH_DOCKER}/example/tmp/{random_id}_input_smiles.csv", index=False)
    
    # Run inference
    args = ArgsInit().args
    args.nclasses = 2
    args.batch_size = 10
    args.use_prot = True
    args.freeze_molecule = True
    args.conv_encode_edge = True
    args.learn_t = True
    args.binary = True
    
    args.use_gpu = True
    args.target = protein_id
    args.target_list = f"{DEFAULT_PATH_DOCKER}/data/datasets/AD/Targets_Fasta.csv"
    args.target_checkpoint_path = f"{DEFAULT_PATH_DOCKER}/example/checkpoints/BINARY_{protein_id}"
    args.input_file_smiles = f"{DEFAULT_PATH_DOCKER}/example/tmp/{random_id}_input_smiles.csv"
    args.output_file = f"{DEFAULT_PATH_DOCKER}/example/tmp/{random_id}_output_predictions.csv"
    
   
    print("Args: ", args)
    main(args)
    
    # Load the CSV file
    df = pd.read_csv(f'{DEFAULT_PATH_DOCKER}/example/tmp/{random_id}_output_predictions.csv')
    
    print("Prediction Results output: ", df)
    return df

def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def run_inference(protein_id, ligand_smile):
    result_df = load_and_filter_data(protein_id, ligand_smile)
    return result_df

def create_interface():
    with gr.Blocks(title="PLA-Net Web Inference") as inference:
        gr.HTML(load_description("gradio/title.md"))

        gr.Markdown("### Input")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Target Protein")
                protein_id = gr.Dropdown(
                    choices=["ada"],
                    label="Target Protein ID",
                    info="Select the target protein from the dropdown menu.",
                    value="ada"
                )
            with gr.Column():
                gr.Markdown("#### Ligand")
                ligand_smile = gr.Textbox(
                    info="Provide SMILES input (separate multiple SMILES with ':' )",
                    placeholder="SMILES input",
                    label="SMILES string(s)",
                )
                gr.Examples(
                    examples=[
                        "Cn4c(CCC(=O)Nc3ccc2ccn(CC[C@H](CO)n1cnc(C(N)=O)c1)c2c3)nc5ccccc45",
                        "OCCCCCn1cnc2C(O)CN=CNc12",
                        "Nc4nc(c1ccco1)c3ncn(C(=O)NCCc2ccccc2)c3n4"
                    ],
                    inputs=ligand_smile,
                    label="Example SMILES"
                )
        btn = gr.Button("Run")
        gr.Markdown("### Output")   
        out = gr.Dataframe(
            headers=["target", "smiles", "interaction_probability", "interaction_class"],
            datatype=["str", "str", "number", "number"],
            label="Prediction Results"
        )
        
        btn.click(fn=run_inference, inputs=[protein_id, ligand_smile], outputs=out)
        
        gr.Markdown("""
                    PLA-Net model for predicting interactions
                    between small organic molecules and one of the 102 target proteins in the AD dataset. Graph representations
                    of the molecule and a given target protein are generated from SMILES and FASTA sequences and are used as
                    input to the Ligand Module (LM) and Protein Module (PM), respectively. Each module comprises a deep GCN
                    followed by an average pooling layer, which extracts relevant features of their corresponding input graph. Both
                    representations are finally concatenated and combined through a fully connected layer to predict the target–
                    ligand interaction probability.
                    """)
        
        gr.Markdown("""
        Ruiz Puentes, P., Rueda-Gensini, L., Valderrama, N. et al. 
        Predicting target–ligand interactions with graph convolutional networks 
        for interpretable pharmaceutical discovery. Sci Rep 12, 8434 (2022). 
        [https://doi.org/10.1038/s41598-022-12180-x](https://doi.org/10.1038/s41598-022-12180-x)
        """)
    
    return inference

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()