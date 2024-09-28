import torch
import numpy as np
import logging
import torch.nn as nn
from gcn_lib.sparse.torch_nn import MLP
from model.model import DeeperGCN

class PLANet(torch.nn.Module):
    def __init__(self, molecule_gcn, target_gcn, hidden_channels, hidden_channels_prot, nclasses, 
                 multi_concat=False, MLP=False, norm=None, saliency=False):
        super(PLANet, self).__init__()

        # Molecule and protein networks
        self.molecule_gcn = molecule_gcn
        self.target_gcn = target_gcn

        # Individual modules' final embedding size
        output_molecule = hidden_channels
        output_protein = hidden_channels_prot
        # Concatenated embedding size
        Final_output = output_molecule + output_protein
        # Overall model's final embedding size
        self.hidden_channels = hidden_channels
        
        self.multi_concat = multi_concat

        # Multiplier
        if multi_concat:
            self.multiplier_prot = torch.nn.Parameter(torch.zeros(hidden_channels))
            self.multiplier_ligand = torch.nn.Parameter(torch.ones(hidden_channels))
        elif MLP:
            # MLP
            hidden_channel = 64
            channels_concat = [256, hidden_channel, hidden_channel, 128]
            self.concatenation_gcn = MLP(channels_concat, norm=norm, last_lin=True)
            # breakpoint()
            indices = np.diag_indices(hidden_channel)
            tensor_linear_layer = torch.zeros(hidden_channel, Final_output)
            tensor_linear_layer[indices[0], indices[1]] = 1
            self.concatenation_gcn[0].weight = torch.nn.Parameter(tensor_linear_layer)
            self.concatenation_gcn[0].bias = torch.nn.Parameter(
                torch.zeros(hidden_channel)
            )
        else:
            # Concatenation Layer
            self.concatenation_gcn = nn.Linear(Final_output, hidden_channels)
            indices = np.diag_indices(output_molecule)
            tensor_linear_layer = torch.zeros(hidden_channels, Final_output)
            tensor_linear_layer[indices[0], indices[1]] = 1
            self.concatenation_gcn.weight = torch.nn.Parameter(tensor_linear_layer)
            self.concatenation_gcn.bias = torch.nn.Parameter(
                torch.zeros(hidden_channels)
            )

        # Classification Layer
        self.num_classes = nclasses
        self.classification = nn.Linear(hidden_channels, nclasses)

    def forward(self, molecule, target):

        molecule_features = self.molecule_gcn(molecule)
        target_features = self.target_gcn(target)
        # print("Molecule features: ", molecule_features)
        # print("Target features: ", target_features)
        # Multiplier
        if self.multi_concat:
            print("Multiplier: ", self.multiplier_prot, self.multiplier_ligand)
            All_features = (
                target_features * self.multiplier_prot
                + molecule_features * self.multiplier_ligand
            )
        else:
            print(molecule_features.shape, target_features.shape)
            
            # Concatenation of LM and PM modules
            All_features = torch.cat((molecule_features, target_features), dim=1)
            # print("Concatenation features: ", All_features)
            All_features = self.concatenation_gcn(All_features)
            #print("Concatenation layer shape: ", All_features.shape)
        # Classification
        classification = self.classification(All_features)
        
        print("Classification features: ", classification)

        return classification

    def print_params(self, epoch=None, final=False):

        logging.info("======= Molecule GCN ========")
        self.molecule_gcn.print_params(epoch)
        logging.info("======= Protein GCN ========")
        self.target_gcn.print_params(epoch)
        if hasattr(self, 'multiplier_prot') and hasattr(self, 'multiplier_ligand'):
            sum_prot_multi = sum(self.multiplier_prot)
            sum_lig_multi = sum(self.multiplier_ligand)
            logging.info("Summed prot multi: {}".format(sum_prot_multi))
            logging.info("Summed lig multi: {}".format(sum_lig_multi))