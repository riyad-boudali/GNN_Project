
#%%
import deepchem.feat
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from tqdm import tqdm
import deepchem 
import os.path as osp

# Version checking
print(f"Torch version: {torch.__version__}")
print(f"Trorch geometric version: {torch_geometric.__version__}")
print(f"Cuda is available: {torch.cuda.is_available()}")


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        """
        raw= The folder where the dataset is should be stored. This
        split into two raw_dir (downloaded data) and processed_dir (processed data)

        """
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        A list of files in the raw_dir which needs to be found in order
        to skip the download.
        """
        return self.filename

    @property
    def processed_file_names(self):
        """
        A list of files in the processed_dir which needs to be found
        in order to skip the processing.
        """
        self.data = pd.read_csv(self.raw_paths[0])
        self.data.index = self.data["index"]
        return [f"data_{i}.pt" for i in list(self.data.index)]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        self.data.index = self.data["index"]
        featurizer = deepchem.feat.MolGraphConvFeaturizer(use_edges= True)
        for index, mol in tqdm(self.data.iterrows(), total= self.data.shape[0]):
            # Featurize moleculs
            f  = featurizer.featurize(mol["smiles"])
            data = f[0].to_pyg_graph()
            data.y = self._get_labels
            data.x = data.x.to(torch.float16)
            data.edge_attr = data.edge_attr.to(torch.float16)
            data.smiles = mol["smiles"]
            torch.save(data, osp.join(self.processed_dir, f'data_{index}.pt'), pickle_protocol=4)
    
    
    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data
    
#%% Test 
import pandas as pd

data = pd.read_csv('data/raw/HIV_train_oversampled.csv')
data.head()


# %%
data = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
# %%
