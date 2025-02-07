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
    failed_idxs = []
    for idx, each_row in df.iterrows():
        mol = MolFromSmiles(each_row["Smiles"])
        molh = AddHs(mol)
        AllChem.EmbedMolecule(molh)
        arr = from_mol_to_array(molh)
        
        # Sometimes get a None for no reason? Embedding should be random. 
        _n = 0
        while arr is None:
            AllChem.EmbedMolecule(molh)
            arr = from_mol_to_array(molh)     
            _n += 1
            if _n > 10:
                print("\tFailed_to_gen_TF3P:", idx)
                with open("Failed_to_gen_TF3P_log.txt", 'a') as f:
                    f.write(str(idx) + "\n")
                    f.write(each_row["Smiles"] + "\n")
                arr = [[0], [0], [[0, 0, 0]]]  # dummy molecule
                failed_idxs.append(idx)

        array_list.append(arr)
    idx_list = df.index.to_list()

    nums_atoms, gs_charge, atom_type, pos = [], [], [], []
    for array, idx in zip(array_list, idx_list):  # see Dataset __getitem__
        # array = self.restore_flat_array(array)  # 从from_mol_to_array() 得来已经是原格式了
        nums_atoms.append(len(array[0]))
        gs_charge += array[0]
        atom_type += array[1]
        pos += array[2]
    return (gs_charge, atom_type, pos, nums_atoms), failed_idxs
    
#%%
def main(df_path: str, output_path: str, model_path="tf3p_trained_models/TF3P-ECFP4-b1024-GS50-W5.pt"):
    df = pd.read_table(df_path, index_col=0)
    array, _ = convert_df_to_array_batch(df)

    model = ForceFieldCapsNet(num_digit_caps=1024)  # more flexibility later
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # change to whatever optimizer was used
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    tensor = model.infer(array)
    arr = tensor.cpu().numpy()
    with open(output_path, 'wb') as f:
        pickle.dump(arr, f)
    

def main_for_topo_project(wd: str, model_path="tf3p_trained_models/TF3P-ECFP4-b1024-GS50-W5.pt"):
    data_txt = [x for x in os.listdir(wd) if x.endswith('txt')]
    assert(len(data_txt) == 1), "Found no or more than one txt file in this folder."
    df = pd.read_table(os.path.join(wd, data_txt[0]), index_col=0)
    
    array_list = []
    failed_list = []
    batch_size = 10
    n_batch = int(len(df) / batch_size) + 1

    device = torch.device("cuda:0")
    for each_df in np.array_split(df, n_batch):
        array, failed = convert_df_to_array_batch(each_df)
    
        model = ForceFieldCapsNet(num_digit_caps=1024)  # more flexibility later
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # change to whatever optimizer was used
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        tensor = model.infer(array)
        arr = tensor.cpu().numpy()
        array_list.append(arr)
        failed_list.extend(failed)
    
    results = np.vstack(array_list)
    for each in failed_list:
        results[df.index.get_loc(each)] = np.nan

    # print(results[:5])
    np.save(os.path.join(wd, "data_TF3P.npy"), results)

#%%
if __name__ == "__main__":
    # Fire(main)
    Fire(main_for_topo_project)

