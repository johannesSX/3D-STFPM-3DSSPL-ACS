import os
import pathlib

import pytorch_lightning as pl

from dataset.dataloader import build_dataset
from utils.stpm import STPM
from run_stpm3D import config_hyperparameters


def run_fmaps_student():
    # Load data with default args (!)
    args_gen, _, args_stpm = config_hyperparameters()

    pl.seed_everything(42, workers=True)

    _, val_dataset, test_dataset = build_dataset(
        args_gen.out_img_size, args_gen.batch_size, args_gen.n_data_workers,
        args_gen.data_dir, args_gen.atlas, args_gen.resample_fac, STPM=True
    )

    # Load model
    CKPT_PATH = "./data/lightning_logs/version_14/checkpoints/epoch=5-val_auroc=0.4719.ckpt"
    model = STPM.load_from_checkpoint(CKPT_PATH)

    save_path = (pathlib.Path(CKPT_PATH).parent).parent / "fmaps"
    os.makedirs(save_path, exist_ok=True)

    # Predict images
    trainer = pl.Trainer(
        logger=False,
        default_root_dir=save_path,
        gpus=-1
    )
    trainer.predict(model, val_dataset)
