"""
Ruibo test inference. 
loading the model and convering mols.

The data loader returns return ((gs_charge, atom_type, pos, nums_atoms), fps), idxs
all of them are lists

原始数据包括 array, fp, idx
array 可以从 from_mol_to_array() 得
转换写在 ZINCH5Dataloader._get_batch
把 array 转化成 (gs_charge, atom_type, pos, nums_atoms)

infer() 输入的array 包括  gs_charge, atom_type, pos, nums_atoms
需要 from_mol_to_array() -> 转换 -> (gs_charge, atom_type, pos, nums_atoms) tuple 格式
"""

import torch
import pandas as pd
import numpy as np
from rdkit.Chem import MolFromSmiles, AddHs, AllChem, DataStructs
from model.networks import ForceFieldCapsNet
from data.force_field import from_mol_to_array

from torch.utils.data import Dataset, DataLoader

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

if __name__ == "__main__":
    
    df = pd.read_table("G:/topological_regression/data/ChEMBL/test_data.txt", index_col=0).iloc[:10]
    # list of field arrays fps
    array_list = []
    fp_list = np.zeros((len(df), 1024))
    i = 0
    for idx, each_row in df.iterrows():
        mol = MolFromSmiles(each_row["Smiles"])
        molh = AddHs(mol)
        AllChem.EmbedMolecule(molh)
        arr = from_mol_to_array(molh)
        array_list.append(arr)
        
        fp2 = AllChem.GetMorganFingerprintAsBitVect(molh, 2, nBits=1024)
        DataStructs.ConvertToNumpyArray(fp2, fp_list[i])

    idx_list = df.index.to_list()
    
    dataset = ChEMBLE_Data(array_list, fp_list, idx_list)


    model = ForceFieldCapsNet(num_digit_caps=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # change to whatever optimizer was used

    checkpoint = torch.load("tf3p_trained_models/TF3P-ECFP4-b1024-GS50-W5.pt")
    model.load_state_dict(checkpoint)


    array_to_infer = dataset.get_array()
    embed = model.infer(array_to_infer)
    

    array_to_infer2 = convert_df_to_array_batch(df)
    embed2 = model.infer(array_to_infer)

