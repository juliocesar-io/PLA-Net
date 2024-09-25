
import os 
import torch
import numpy as np
from tqdm import tqdm
from data.dataset import transform_molecule_pg
import pandas as pd
import torch.nn.functional as F
import torch

def load_model(model, fold, args):
    model_name = os.path.join(args.target_checkpoint_path, f'Fold{fold}','Best_Model.pth')
        
    pre_model = torch.load(model_name,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(pre_model['model_state_dict'])

    return model   

@torch.no_grad()
def test_gcn(model, device, loader,args):
    first = True
    
    for batch in tqdm(loader, desc="Iteration"):
        save_dict = {'Target': [],
                 'Smiles': [],
                 'Probability of Interaction': [],
                 'Class Id': []}
        save_dict_temp = {
                 'Folder 1': [],
                 'Folder 2': [],
                 'Folder 3': [],
                 'Folder 4': []}

        if args.use_prot:
            batch_mol = batch[0].to(device)
            batch_prot = batch[1].to(device)
            smiles = batch_mol['smiles']
            smiles = [smi for smi in smiles]
        else:
            batch_mol = batch[0].to(device)
            smiles = batch_mol['y']
            smiles = [smi for smi in smiles]
            
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            # only retain the top two node/edge features
            num_features = args.num_features
            batch_mol.x = batch_mol.x[:, :num_features]
            batch_mol.edge_attr = batch_mol.edge_attr[:, :num_features]
        if batch_mol.x.shape[0] == 1:
            pass
        else:

            target = [args.target]*len(batch[0].y)
            save_dict['Target'].extend(target)
            save_dict['Smiles'].extend(smiles)
            for fold in range(1,5):
                model = load_model(model, fold, args)
                model.eval()

                with torch.set_grad_enabled(False):   
                    if args.use_prot:
                        pred = model(batch_mol,batch_prot)
                    else:
                        pred = model(batch_mol)
                    pred = F.softmax(pred,dim=1)
                    save_dict_temp[f'Folder {fold}'].extend(pred.cpu().tolist()) 
            for fold in range(1,5):
                save_dict_temp[f'Folder {fold}'] = np.array(save_dict_temp[f'Folder {fold}'])

            save_dict['Probability of Interaction'] = np.mean([save_dict_temp['Folder 1'], save_dict_temp['Folder 2'], save_dict_temp['Folder 3'], save_dict_temp['Folder 4']], axis = 0).tolist()
            save_dict['Class Id'] = [int(np.argmax(i)) for i in save_dict['Probability of Interaction']]
            save_dict['Probability of Interaction'] = [x[1] for x in save_dict['Probability of Interaction']]
            for fold in range(1,5):
                save_dict_temp[f'Folder {fold}'] = save_dict_temp[f'Folder {fold}'].tolist()
            
            save_df = pd.DataFrame(save_dict)

            save_path = os.path.join(args.output_file)
            if first == 0:
                save_df.to_csv(save_path, index=False)
                first = False
            else:
                save_df.to_csv(save_path, mode='a', header=False, index= False)
                
                
                
def get_dataset_inference(
    dataset, use_prot=False, target=None, args=None, advs=False, saliency=False
):
    DEFAULT_LABEL = 0
    total_dataset = []
    if use_prot:
        prot_graph = transform_molecule_pg(
            target["Fasta"].item(), label=None, is_prot=use_prot
        )

    for mol, label in tqdm(
        zip(dataset["Smiles"], [DEFAULT_LABEL]*len(dataset["Smiles"])), total=len(dataset["Smiles"])
    ):
        if use_prot:
            total_dataset.append(
                [
                    transform_molecule_pg(mol, label, args, advs, saliency=saliency),
                    prot_graph,
                ]
            )
        else:
            total_dataset.append(
                transform_molecule_pg(mol, label, args, advs, saliency=saliency)
            )
    return total_dataset
    