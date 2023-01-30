import pandas as pd
import torch
from torch.utils.data import Dataset

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self,file_name, label_col):
        super().__init__()
        self.label_col=label_col
        self.data=pd.read_csv(file_name).select_dtypes(exclude=[object])
        column_to_move = self.data.pop(label_col)
        self.data.insert(0, label_col, column_to_move)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example=self.data.iloc[idx]
        features=torch.tensor(example[1:10])
        label=example[0]
        return features, label

