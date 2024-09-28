import os 
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

def load_model(model, fold, target_checkpoint_path):
    model_name = os.path.join(target_checkpoint_path, f'Fold{fold}', 'Best_Model.pth')
    print(f"model_name: {model_name}")
    pre_model = torch.load(
        model_name,
        map_location=lambda storage, loc: storage
    )
    model.load_state_dict(pre_model['model_state_dict']) 
    return model   

@torch.no_grad()
def run_inference_model_gcn(
    model,
    device,
    loader,
    use_prot,
    feature,
    num_features,
    target,
    output_file,
    target_checkpoint_path
):
    for batch in tqdm(loader, desc="Iteration"):
        save_dict = {
            'target': [],
            'smiles': [],
            'interaction_probability': [],
            'interaction_class': []
        }
        save_dict_temp = {
            'Folder 1': [],
            'Folder 2': [],
            'Folder 3': [],
            'Folder 4': []
        }

        if use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
            smiles = batch_mol['smiles']
            smiles = [smi for smi in smiles]
        else:
            batch_mol = batch[0].to(device)
            smiles = batch_mol['y']
            smiles = [smi for smi in smiles]
            
        if feature == 'full':
            pass
        elif feature == 'simple':
            # Only retain the top two node/edge features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        
        if batch_mol.x.shape[0] == 1:
            pass
        else:
            target_list = [target] * len(batch[0].y)
            save_dict['target'].extend(target_list)
            save_dict['smiles'].extend(smiles)
            for fold in range(1, 5):
                model = load_model(model, fold, target_checkpoint_path)
                model.eval()
                if use_prot:
                    pred = model(batch_mol, batch_prot)
                else:
                    pred = model(batch_mol)
                pred = F.softmax(pred, dim=1)
                save_dict_temp[f'Folder {fold}'].extend(pred.cpu().tolist()) 
            
            for fold in range(1, 5):
                save_dict_temp[f'Folder {fold}'] = np.array(save_dict_temp[f'Folder {fold}'])

            probabilities = np.mean(
                [
                    save_dict_temp['Folder 1'],
                    save_dict_temp['Folder 2'],
                    save_dict_temp['Folder 3'],
                    save_dict_temp['Folder 4']
                ],
                axis=0
            ).tolist()
            save_dict['interaction_probability'] = [x[1] for x in probabilities]
            save_dict['interaction_class'] = [int(np.argmax(i)) for i in probabilities]
            
            save_df = pd.DataFrame(save_dict)
            save_path = os.path.join(output_file)
            print("Saving results to CSV file:", save_path)
            save_df.to_csv(save_path, mode='a', header=True, index=False)