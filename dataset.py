# Imported packages
import torch
import torch_geometric
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem

# Version check
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


# Creating custom dataset
class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
    @property
    def raw_file_names(self):
        """
        A list of files in the raw_dir which needs
        to be found in order to skip the download.
        """
        return self.filename

    @property
    def processed_file_names(self):
        """
        A list of files in the processed_dir which
        needs to be found in order to skip the processing.
        """
        
        self.data = pd.read_csv(self.raw_paths[0])
        self.data.index = self.data["index"]
        return [f"data_{i}.pt" for i in list(self.data.index)]


    def download(self):
        # skipping the exection of the download function
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            if mol_obj is None:
                print(f"Invalid SMILES string at index {index}: {mol['smiles']}")
                continue  # Skip this molecule if it is invalid
            # Get node features
            node_feats = self._get_node_features(mol=mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol=mol_obj)
            # Get adjancy info
            edge_index = self._get_adjacency_info(mol=mol_obj)
            # Get labels
            label = self._get_labels(mol["HIV_active"])

            # Create data object
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=label,
                smiles=mol["smiles"],
            )
            torch.save(data, osp.join(self.processed_dir, f"data_{index}.pt"))

    def _get_node_features(self, mol):
        """
        Node feature matrix with shape [num_nodes, num_node_features]
        """

        all_node_features = []
        for atom in mol.GetAtoms():
            node_feats = []
            # Feature1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature2: Atomic Degree
            node_feats.append(atom.GetDegree())
            # Feature3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature4: Hyberidization
            node_feats.append(atom.GetHybridization())
            # Feature5: Number of H
            node_feats.append(atom.GetTotalNumHs())
            # Feature6: Number of radical e
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature7: Chirality
            node_feats.append(atom.GetChiralTag())
            # Feature8: is aromatic
            node_feats.append(atom.GetIsAromatic())
            # Feature9: is in ring
            node_feats.append(atom.IsInRing())
            all_node_features.append(node_feats)

        all_node_features = np.asarray(all_node_features)
        return torch.tensor(all_node_features, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        Edge feature matrix with shape [num_edges, num_edge_features]
        """
        all_edge_featuers = []
        for bond in mol.GetBonds():
            edge_feats = []
            # Feature1: Bond type
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature2: Stereo configuration
            edge_feats.append(bond.GetStereo())
            # Feaure3: Is conjucted
            edge_feats.append(bond.GetIsConjugated())
            all_edge_featuers.append(edge_feats)

        all_edge_featuers = np.asarray(all_edge_featuers)
        return torch.tensor(all_edge_featuers, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data
    
#%%
train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
# %%
