"""
Ruibo inference.
generate embedding, 

暂时用pickle 存结果。chembl 的也不太大
"""

import os
import torch
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles, AddHs, AllChem, DataStructs
from model.networks import ForceFieldCapsNet
from data.force_field import from_mol_to_array

from torch.utils.data import Dataset, DataLoader

from fire import Fire
import pickle
#%%
class ChEMBLE_Data(Dataset):
    def __init__(self, batch_array, batch_fp, batch_idx):
        """
        Parameters
        ----------
        batch_array : iterable
            DESCRIPTION.
        batch_fp : iterable
            DESCRIPTION.
        batch_idx : iterable
            DESCRIPTION.

        Returns
        -------
        None.

        """
        idxs, fps = [], []
        nums_atoms, gs_charge, atom_type, pos = [], [], [], []
        for array, fp, idx in zip(batch_array, batch_fp, batch_idx):  # see Dataset __getitem__
            # array = self.restore_flat_array(array)  # 从from_mol_to_array() 得来已经是原格式了
            nums_atoms.append(len(array[0]))
            gs_charge += array[0]
            atom_type += array[1]
            pos += array[2]
            fps.append(fp.tolist())  # array to list
            # idxs.append(int(idx))  # np.int to int  # 现在用 str index
            idxs.append(idx)
        self.data = ( ((gs_charge, atom_type, pos, nums_atoms), fps), idxs )


    def __len__(self):
        return len(self.fp)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    # @staticmethod
    # def restore_flat_array(fa):
    #     fa = fa.reshape(-1, 5)
    #     return (fa[:, 0].tolist(),
    #             fa[:, 1].astype('int').tolist(),
    #             fa[:, 2:].tolist())
    
    def get_array(self):
        return self.data[0][0]

#%%

def convert_df_to_array_batch(df: pd.DataFrame) -> tuple:
    # 可能不需要一批一批处理，应该是单个的，但一批的可以用，不改了。
    array_list = []
    for idx, each_row in df.iterrows():
        mol = MolFromSmiles(each_row["Smiles"])
        molh = AddHs(mol)
        AllChem.EmbedMolecule(molh)
        arr = from_mol_to_array(molh)
        array_list.append(arr)
    idx_list = df.index.to_list()

    nums_atoms, gs_charge, atom_type, pos = [], [], [], []
    for array, idx in zip(array_list, idx_list):  # see Dataset __getitem__
        # array = self.restore_flat_array(array)  # 从from_mol_to_array() 得来已经是原格式了
        nums_atoms.append(len(array[0]))
        gs_charge += array[0]
        atom_type += array[1]
        pos += array[2]
    return (gs_charge, atom_type, pos, nums_atoms)
    
#%%
def main(df_path: str, output_path: str):
    df = pd.read_table(df_path, index_col=0)
    tensor = convert_df_to_array_batch(df)
    arr = tensor.cpu().numpy()
    with open(output_path, 'wb') as f:
        pickle.dump(arr, f)
    

def main_for_topo_project(wd: str):
    data_txt = [x for x in os.listdir(wd) if x.endswith('txt')]
    assert(len(data_txt) == 1), "Found no or more than one txt file in this folder."
    df = pd.read_table(os.path.join(wd, data_txt[0]), index_col=0)
    tensor = convert_df_to_array_batch(df)
    arr = tensor.cpu().numpy()
    np.save(os.path.join(wd, "data_TF3P.npy"), arr)
    
#%%
if __name__ == "__main__":
    # Fire(main)
    Fire(main_for_topo_project)

