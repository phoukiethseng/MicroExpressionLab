import torch
from torch.utils.data import Dataset

class MicroExpressionDataset(Dataset):
    def __init__(self, label_file_path, dataset_path):
