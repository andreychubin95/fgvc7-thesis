import os
import typing
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class PlantsDataset(Dataset):
    def __init__(self, folder: str, df: pd.DataFrame, transform: typing.Callable = None):
        self.path = folder
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        path = os.path.join(self.path, self.df.iloc[index]['path'])
        label = torch.from_numpy(self.df.drop(['path', 'image_id'], axis=1).iloc[index].values)
        image = read_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
