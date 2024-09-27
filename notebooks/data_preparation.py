import numpy as np
from rdkit import Chem
from data.features import (
    atom_to_feature_vector,
    bond_to_feature_vector
)

def smiles_to_graph(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    return mol_to_graph(mol)

def protein_to_graph(sequence_aa_string):
    mol = Chem.MolFromFASTA(sequence_aa_string)
    return mol_to_graph(mol)

def mol_to_graph(mol):
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        ftrs = atom_to_feature_vector(atom)
        atom_features_list.append(ftrs)

    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    return edge_attr, edge_index, x