import os
import gc
import lightning as pl
import numpy as np
import pandas as pd
import albumentations as albm
from albumentations.pytorch import ToTensorV2 as ToTensor

from .utils import get_loaders, get_test_dataloader
from .custom_callbacks import progress_bar, MetricsCallback, checkpoint_callback


class Experiment(object):
    def __init__(
        self,
        folder: str,
        batch_size: int,
        model: pl.LightningModule,
        filename: str,
        height: int = 224,
        width: int = 224
    ):
        self.folder = folder
        self.batch_size = batch_size
        self.filename = filename

        if not os.path.exists(f'{self.filename}_logs'):
            os.makedirs(f'{self.filename}_logs')

        self._train_transforms = albm.Compose([
            albm.Resize(height=height+32, width=width+32),
            albm.RandomCrop(height=height, width=width),
            albm.HorizontalFlip(p=0.5),
            albm.VerticalFlip(p=0.5),
            albm.ShiftScaleRotate(rotate_limit=25.0, p=0.7),
            albm.OneOf([
                albm.IAAEmboss(p=1),
                albm.IAASharpen(p=1),
                albm.Blur(p=1)], p=0.5),
            albm.IAAPiecewiseAffine(p=0.5),
            albm.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
            ToTensor()
        ])

        self._test_transforms = albm.Compose([
            albm.Resize(height=height, width=width),
            albm.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
            ToTensor()
        ])

        self._train_data = pd.read_csv(f'{self.folder.split("/")[0]}/corrected_train.csv')

        self.model = model
        self.trainer = None

    def create_submission(self) -> None:
        test = pd.read_csv(f'{self.folder.split("/")[0]}/test.csv')
        test['path'] = test['image_id'] + '.jpg'
        test_loader = get_test_dataloader(self.folder, test, self._test_transforms, self.batch_size)
        preds = self.trainer.predict(self.model, test_loader, ckpt_path='best', return_predictions=True)
        preds = np.concatenate([x.numpy() for x in preds])
        cols = [x for x in self._train_data.columns if x not in ['image_id', 'path']]
        res = pd.DataFrame(preds, columns=cols, index=test.image_id.values) \
            .reset_index(drop=False) \
            .rename(columns={'index': 'image_id'})
        res.to_csv(f'subs/{self.filename}.csv', index=False)

    @staticmethod
    def _to_item(dict_: dict) -> dict:
        for x in dict_.keys():
            dict_[x] = dict_[x].item()
        return dict_

    def save_history(self) -> None:
        data = pd.DataFrame([self._to_item(x) for x in self.trainer.callbacks[1].metrics])
        data = data.reset_index(drop=False)
        data.to_csv(f'history/{self.filename}.csv', index=False)

    def run(self) -> None:
        train_loader, val_loader = get_loaders(
            self.folder, self._train_data, self._train_transforms, self._test_transforms, self.batch_size
        )

        pb = progress_bar()
        ck = checkpoint_callback()

        self.trainer = pl.Trainer(
                default_root_dir=f'{self.filename}_logs',
                min_epochs=70,
                max_epochs=100,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
                callbacks=[pb, MetricsCallback(), ck],
                deterministic=False
            )

        self.trainer.fit(self.model, train_loader, val_loader)
        _ = gc.collect()

        print(ck.best_model_path)
