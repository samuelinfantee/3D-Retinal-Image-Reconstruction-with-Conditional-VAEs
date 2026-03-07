import glob
import os
import random
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset
def get_train_val_test(root_dir: os.path, split_ratios: Tuple[float], seed: int):
    
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    # Grab all the subfolders 
    subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d))]
    
    if not subfolders:
        raise RuntimeError(f"No subfolders found in {root_dir}")
    random.seed(seed)
    random.shuffle(subfolders)
    # Calculate number of volumes in each split
    n_folders = len(subfolders)
    n_train = int(n_folders * split_ratios[0])
    n_val = int(n_folders * split_ratios[1])
    # Slice the folders list
    train_folders = subfolders[:n_train]
    val_folders = subfolders[n_train : n_train + n_val]
    test_folders = subfolders[n_train + n_val:]
    
    print(f"Split Summary: {len(train_folders)} train volumes, "
          f"{len(val_folders)} eval volumes, {len(test_folders)} test volumes")
    #  Gather all .npz files from a list of folders
    def gather_files(folder_list):
        files = []
        for folder in folder_list:
            # Recursive search inside each specific folder
            files.extend(glob.glob(os.path.join(folder, '**', '*.npz'), recursive=True))
        return files

    return gather_files(train_folders), gather_files(val_folders), gather_files(test_folders)

class OCTA500SplitDataset(Dataset):
    def __init__(self, files_list: List[os.PathLike], sampling_rate: float, 
        include_sampl_mat: bool, transform=None):
        
        self.files_list = files_list
        self.sampling_rate = sampling_rate
        self.include_samp_mat = include_sampl_mat
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.files_list)

    def __getitem__(self, idx):
        # Load a sample from the dataset at a given index
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the file path
        file_path = self.files_list[idx]
        # Load the .npz file
        try:
            with np.load(file_path) as data:
                # Extract the specific arrays you need
                # Ensure they are copied/converted to avoid "negative stride" issues with PyTorch
                input_data = data["arr_0"].copy()
        
        except KeyError as e:
            raise KeyError(f"Dataset element {e} not found in {file_path}.")
        # Apply Transforms (if any)
        if self.transform:
            input_data = self.transform(input_data)
        # Convert to a tensor
        input_data = torch.from_numpy(input_data).float().unsqueeze(0)
        # Create a volumetric sampling mask and apply it
        mask = (torch.rand_like(input_data) < self.sampling_rate).float()
        masked_input = input_data * mask

        if self.include_samp_mat:
            masked_input = torch.cat([masked_input, mask], dim=0)
        
        return masked_input, input_data
