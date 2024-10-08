import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from config import EXP_STATE_SIZE, EXP_CLASS_SIZE

class MicroExpressionDataset(Dataset):
    def __init__(self, label_file_path, dataset_path):
        self.label = pd.read_csv(label_file_path, header=0, index_col=False)
        self.dataset_path = dataset_path
        self.resizer = transforms.Resize((64, 64))
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.label.iloc[idx, 0])
        imgs = self.resizer(read_image(file_path))
        exp_class_indices = torch.tensor(self.label.iloc[idx, 1], dtype=torch.int32)
        exp_state_indices = torch.tensor(self.label.iloc[idx, 2], dtype=torch.int32)

        exp_class_label = torch.zeros(EXP_CLASS_SIZE, dtype=torch.int32)
        exp_state_label = torch.zeros(EXP_STATE_SIZE, dtype=torch.int32)

        exp_class_label[exp_class_indices - 1] = 1
        exp_state_label[exp_state_indices - 1] = 1

        return imgs, exp_class_label, exp_state_label