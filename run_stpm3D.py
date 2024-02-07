import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.dataloader import build_dataset
from model.patch_learning import ResNetClassifier
from utils.stpm import STPM


logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def config_hyperparameters():
    # Data generator params
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, choices=['pretrain', 'train', 'eval', 'fmaps'], default='eval')
    parser.add_argument("--batch_size", type=int, default=2)  # 8
    parser.add_argument("--out_img_size", type=int, default=[156, 156, 156])  # [144, 176, 128]
    parser.add_argument("--resample_fac", type=float, default=39/56) # 1
    parser.add_argument("--n_data_workers", type=int, default=0) # 10
    parser.add_argument("--data_dir", type=str, default="./data/{}")  # 2022_09_12_STPM3D
    parser.add_argument("--net_type", type=str, choices=["RESNET", "CONVNET"], default="RESNET")
    parser.add_argument("--num_classes", type=int, default=128) # 128
    parser.add_argument("--resnet_version", type=int, choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument("--convnet_version", type=str, choices=['tiny', 'small', 'base', 'large'], default='tiny')
    parser.add_argument("--atlas", type=str2bool, default=False)

    args_gen, _ = parser.parse_known_args()

    # Teacher training params (Patch Learning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=64) # 250
    parser.add_argument("--pretrained", type=str2bool, default=True)  # 250
    parser.add_argument("--loss_type", type=str, choices=["TRI", "TRI2", "SUP"], default="TRI") # TRI
    args_resnet, _ = parser.parse_known_args()

    # STPM training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=64)  # 150
    # parser.add_argument("--ckpt_path", default=None) #default="../results/2022_09_12_STPM3D/lightning_logs/version_10/checkpoints/last_epoch_teacher.ckpt", type=str)
    parser.add_argument("--ckpt_path", default="./data/lightning_logs/version_11248/checkpoints/last_epoch_teacher.ckpt", type=str)
    parser.add_argument('--amap_mode', choices=['mul', 'sum'], default='mul')
    parser.add_argument("--in_channels", type=int, default=1)
    args_stpm, _ = parser.parse_known_args()

    return args_gen, args_resnet, args_stpm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_stpm(args_gen, args_stpm):
    print("args_gen: {} ".format(args_gen))
    print("args_stpm: {}".format(args_stpm))

    pl.seed_everything(42, workers=True)

    train_dataset, val_dataset, test_dataset = build_dataset(
        args_gen.out_img_size, args_gen.batch_size, args_gen.n_data_workers,
        args_gen.data_dir, args_gen.atlas, args_gen.resample_fac, STPM=True
    )

    model = STPM(**vars(args_stpm))

    checkpoint_1 = ModelCheckpoint(
        filename='{epoch}-{val_auroc:.4f}',
        monitor="val_auroc", #"val_loss", #'val_pixel_auc',
        mode='max',
        save_top_k=10,
    )
    checkpoint_2 = ModelCheckpoint(
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=10,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        default_root_dir=args_gen.data_dir.format(""),
        callbacks=[checkpoint_1, checkpoint_2],
        # check_val_every_n_epoch=1,#20
        # default_root_dir=args_stpm.path_log_f.format(""),
        log_every_n_steps=1,
        devices=[0], #[1, 2], #[1, 2],
        max_epochs=args_stpm.n_epochs,
        # strategy='ddp',
        num_sanity_val_steps=2,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataset,
        val_dataloaders=val_dataset
    )
    trainer.save_checkpoint(os.path.join(trainer.log_dir, "checkpoints", "last_epoch_student.ckpt"))


def run_p_resnet(args_gen, args_resnet, args_stpm):
    print("args_gen: {} ".format(args_gen))
    print("args_resnet: {}".format(args_resnet))
    print("args_stpm: {}".format(args_stpm))

    pl.seed_everything(42, workers=True)

    train_dataset, val_dataset, test_dataset = build_dataset(
        args_gen.out_img_size, args_gen.batch_size, args_gen.n_data_workers,
        args_gen.data_dir, args_gen.atlas, args_gen.resample_fac, RESNET=True
    )

    if args_gen.net_type == 'RESNET':
        version = args_gen.resnet_version
    elif args_gen.net_type == 'CONVNET':
        version = args_gen.convnet_version
    model_resnetcl = ResNetClassifier(
        num_classes=args_gen.num_classes,
        in_channels=args_stpm.in_channels,
        net_type=args_gen.net_type,
        loss_type=args_resnet.loss_type,
        version=version,
        atlas=args_gen.atlas,
        tiles=True,
        pretrained=args_resnet.pretrained
    )
    logger.debug("ResNet Model built.")

    checkpoint_1 = ModelCheckpoint(
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=5,
        mode='min',
    )

    checkpoint_2 = ModelCheckpoint(
        filename='{epoch}-{train_loss:.4f}',
        monitor='train_loss',
        save_top_k=5,
        mode='min',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        default_root_dir=args_gen.data_dir.format(""),
        callbacks=[checkpoint_1, checkpoint_2],
        log_every_n_steps=1,
        devices=[0],
        max_epochs=args_resnet.n_epochs,
        strategy='ddp',
    )
    logger.debug("Trainer built.")

    trainer.fit(
        model=model_resnetcl,
        train_dataloaders=train_dataset,
        val_dataloaders=val_dataset
    )
    trainer.save_checkpoint(os.path.join(trainer.log_dir, "checkpoints", "last_epoch_teacher.ckpt"))


if __name__ == "__main__":
    args_gen, args_resnet, args_stpm = config_hyperparameters()

    if args_gen.run_mode == 'pretrain':
        run_p_resnet(args_gen, args_resnet, args_stpm)
    elif  args_gen.run_mode == 'train':
        run_stpm(args_gen, args_stpm)
    elif args_gen.run_mode == 'eval':
        from model.eval import run_eval
        run_eval(KMEANS=False, MODE="CLASS", version=11249, ckpt='epoch=63-val_auroc=0.8710.ckpt')
        run_eval(KMEANS=False, MODE="SEG", version=11249, ckpt='epoch=63-val_auroc=0.8710.ckpt')
    elif args_gen.run_mode == 'fmaps':
        from model.fmaps import run_fmaps_student
        run_fmaps_student()

