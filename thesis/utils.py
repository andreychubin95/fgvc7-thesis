import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .torch_dataset import PlantsDataset, PlantsDatasetTest


def _split(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    frame = data.drop('path', axis=1) \
        .set_index('image_id') \
        .idxmax(axis=1) \
        .reset_index(drop=False) \
        .rename(columns={0: 'target'})

    train, val = train_test_split(frame, test_size=0.20, stratify=frame['target'], random_state=42)
    return data.loc[train.index], data.loc[val.index]


def get_loaders(
        folder: str, train_data: pd.DataFrame, train_transforms: Compose, test_transforms: Compose, batch_size: int
):
    train_df, val_df = _split(train_data)

    train_dataset = PlantsDataset(folder, train_df, transform=train_transforms)
    val_dataset = PlantsDataset(folder, val_df, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False, num_workers=4)

    return train_loader, val_loader


def get_test_data(path: str, data: pd.DataFrame, transform: Compose) -> torch.Tensor:
    tensor_list = []

    for filename in tqdm(data['path'].values):
        full_path = os.path.join(path, filename)
        image = np.array(Image.open(full_path).convert('RGB'))
        if image.shape != (1365, 2048, 3):
            image = np.swapaxes(image, 0, 1)
        image = transform(image=image)['image']
        tensor_list.append(image)

    return torch.stack(tensor_list)


def get_test_dataloader(path: str, data: pd.DataFrame, transform: Compose, batch_size: int) -> DataLoader:
    dataset = PlantsDatasetTest(path, data, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size//2, shuffle=False, num_workers=4)
    return dataloader


def predict_fn(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        y_hat = model(X)
    return y_hat.cpu().numpy()
