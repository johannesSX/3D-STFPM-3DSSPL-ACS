from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import AUROC, ROC, AveragePrecision, Dice, JaccardIndex
from torch import Tensor

from utils.utils import remove_all_forward_hooks
from model.patch_learning import build_resnet, build_convnet, ResNetClassifier


class STPM(pl.LightningModule):
    def __init__(self,
                 in_channels=2,
                 resnet_version=18,
                 net_type='RESNET',
                 ckpt_path=None,
                 amap_mode="mul",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.amap_mode = amap_mode
        self.lr = 0.5

        if ckpt_path is not None:
            # Build teacher
            # ckpt_path = '/media/johsch/newtondata/phd_3DSTFPM/data/lightning_logs/version_11248/checkpoints/last_epoch_teacher.ckpt'
            _teacher = ResNetClassifier.load_from_checkpoint(ckpt_path)
            version = _teacher.hparams["version"]
            self.teacher = _teacher.resnet_model
            remove_all_forward_hooks(self.teacher)

            # Freeze teacher
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = build_resnet(in_channels=in_channels, resnet_version=resnet_version, pretrained=False)
            for param in self.teacher.parameters():
                param.requires_grad = False
            version = 18
            self.teacher.eval()

        # Build student
        if net_type == "RESNET":
             self.student = build_resnet(in_channels=in_channels, resnet_version=version, pretrained=False)
             # self.student = build_resnet(in_channels=in_channels, resnet_version=resnet_version, pretrained=False)
        elif net_type == "CONVNET":
            self.student = build_convnet(in_channels=in_channels, convnet_version=version, pretrained=False)

        # Hooks for intermediate feature map output during forward pass
        self.init_features()

        def hook_t(module, input, output):
            fmap = F.normalize(output, p=2, dim=1)
            self.features_t.append(fmap)

        def hook_s(module, input, output):
            fmap = F.normalize(output, p=2, dim=1)
            self.features_s.append(fmap)

        if net_type == "RESNET":
            self.teacher.layer1[-1].register_forward_hook(hook_t)
            self.teacher.layer2[-1].register_forward_hook(hook_t)
            self.teacher.layer3[-1].register_forward_hook(hook_t)
            self.student.layer1[-1].register_forward_hook(hook_s)
            self.student.layer2[-1].register_forward_hook(hook_s)
            self.student.layer3[-1].register_forward_hook(hook_s)
        elif net_type == "CONVNET":
            self.teacher.stages[0][-1].register_forward_hook(hook_t)
            self.teacher.stages[1][-1].register_forward_hook(hook_t)
            self.teacher.stages[2][-1].register_forward_hook(hook_t)
            self.student.stages[0][-1].register_forward_hook(hook_t)
            self.student.stages[1][-1].register_forward_hook(hook_t)
            self.student.stages[2][-1].register_forward_hook(hook_t)

        # Metrics
        # Detection
        self.auroc = AUROC(task='binary')
        self.roc = ROC(task='binary')
        self.avg_prec = AveragePrecision(task='binary')
        # Segmentation
        self.seg_iou = JaccardIndex(task='binary', num_classes=2, threshold=0.01)
        self.seg_dice = Dice(threshold=0.01)
        self.seg_auroc = AUROC(task='binary')
        self.seg_roc = ROC(task='binary', compute_on_cpu=True)

        self.seg_roc = self.seg_roc.cpu()

    def init_features(self):
        self.features_t = []
        self.features_s = []

    def forward(self, x):
        self.init_features()
        self.teacher.eval()

        # Call hooks with forward pass
        with torch.no_grad():
            self.teacher(x)
        self.student(x)

        return self.features_t, self.features_s

    def loss_function(self, ft_list: Tensor, fs_list: Tensor) -> Tensor:
        loss = 0
        for ft_norm, fs_norm in zip(ft_list, fs_list):
            # _, _, h, w = fs_norm.shape
            # f_loss = (0.5 / (w * h)) * F.mse_loss(fs_norm, ft_norm, reduction="sum")
            # loss += f_loss

            # Paper: 0.5 * torch.linalg.vector_norm(fs_norm - ft_norm, dim=1)
            # Proportinal to: 0.5 * (1. - F.cosine_similarity(fs_norm, ft_norm, dim=1))
            f_loss: Tensor = 0.5 * torch.linalg.vector_norm(ft_norm - fs_norm, dim=1)
            loss += f_loss.mean(dim=(1, 2, 3))

        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, _, _ = batch

        features_t, features_s = self(x)
        loss = self.loss_function(features_t, features_s)

        # Metrics
        batch_size = len(x)
        self.log('train_loss', loss, batch_size=batch_size)

        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int):
        x, label, mask = batch

        # Inference
        features_t, features_s = self(x)
        loss = self.loss_function(features_t, features_s)

        anomaly_map, amaps = self.anomaly_map(features_t, features_s, x.shape[-3:])
        score = anomaly_map.amax((1, 2, 3, 4))

        # Metrics
        batch_size = len(x)
        self.log("val_loss", loss, batch_size=batch_size)
        self.auroc.update(score, label)
        self.log("val_auroc", self.auroc)
        self.avg_prec.update(score, label)
        self.log("val_avg_prec", self.avg_prec)
        self.roc.update(score, label)  # no logging
        self.log("hp_metric", self.auroc)  # for TensorBoard visualization

    def on_validation_epoch_end(self):
        self.roc.reset()
        self.seg_roc.reset()

    def configure_optimizers(self):
        # TODO: Optimize parameters
        return torch.optim.SGD(self.student.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)

    #@torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, label, mask = batch

        # Inference
        features_t, features_s = self(x)
        loss = self.loss_function(features_t, features_s)

        batch_size = len(x)
        self.log("val_loss", loss, batch_size=batch_size)

        anomaly_map, amaps = self.anomaly_map(features_t, features_s, x.shape[-3:])
        score = anomaly_map.amax((1, 2, 3, 4))

        self.auroc.cpu()
        self.avg_prec.cpu()
        self.roc.cpu()

        self.seg_iou.cpu()
        self.seg_dice.cpu()
        self.seg_auroc.cpu()

        self.auroc.update(score.cpu(), label.cpu())
        self.avg_prec.update(score.cpu(), label.cpu())
        self.roc.update(score.cpu(), label.cpu())  # no logging


    def on_test_epoch_end(self):
        # Visualize and save ROC
        fpr, tpr, threshs = self.roc.compute()
        self.roc.reset()

        auroc = self.auroc.compute().cpu().numpy()
        self.log("val_auroc", self.auroc.compute().cpu())
        self.auroc.reset()

        fpr, tpr, threshs = fpr.cpu().numpy(), tpr.cpu().numpy(), threshs.cpu().numpy()
        self.save_roc(fpr, tpr, threshs, auroc)

        del fpr, tpr, threshs
        self.log("val_avg_prec", self.avg_prec)



    @torch.no_grad()
    def anomaly_map(self,
                    ft_list: List[Tensor],
                    fs_list: List[Tensor],
                    out_size: Tuple[int, int, int]) -> Tuple[Tensor, Tensor]:
        # if self.amap_mode == 'mul':
        # 	anomaly_map = np.ones([out_size, out_size])
        # else:
        # 	anomaly_map = np.zeros([out_size, out_size])

        # Calculate anomaly maps from loss of feature maps (paper 3.3)
        amaps = []
        for ft_norm, fs_norm in zip(ft_list, fs_list):
            # Recalculate loss
            amap = 0.5 * torch.linalg.vector_norm(ft_norm - fs_norm, dim=1, keepdim=True)

            # fs_norm = F.normalize(fs, p=2)
            # ft_norm = F.normalize(ft, p=2)
            # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            # a_map = torch.unsqueeze(a_map, dim=1)
            amap = F.interpolate(amap, size=out_size, mode='trilinear', align_corners=False)
            # amap = amap[0, 0, :, :].to('cpu').detach().numpy()
            amaps.append(amap)

        # Reduce anomaly maps to one (paper eq. 4)
        anomaly_map = torch.stack(amaps)
        if self.amap_mode == 'mul':
            anomaly_map = torch.prod(anomaly_map, dim=0)
        else:
            anomaly_map = torch.sum(anomaly_map, dim=0)

        return anomaly_map, amaps