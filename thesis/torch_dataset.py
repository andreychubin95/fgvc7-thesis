import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from albumentations.core.composition import Compose


class PlantsDataset(Dataset):
    def __init__(self, folder: str, df: pd.DataFrame, transform: Compose = None):
        self.path = folder
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        path = os.path.join(self.path, self.df.iloc[index]['path'])
        label = torch.tensor(np.argmax(self.df.drop(['path', 'image_id'], axis=1).iloc[index].values))
        image = np.array(Image.open(path).convert('RGB'))
        if image.shape != (1365, 2048, 3):
            image = np.swapaxes(image, 0, 1)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, label


class PlantsDatasetTest(PlantsDataset):
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor):
        path = os.path.join(self.path, self.df.iloc[index]['path'])
        image = np.array(Image.open(path).convert('RGB'))
        if image.shape != (1365, 2048, 3):
            image = np.swapaxes(image, 0, 1)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image
