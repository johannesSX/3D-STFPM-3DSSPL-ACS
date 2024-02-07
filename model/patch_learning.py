import os
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from acsconv.converters import ACSConverter
from acsconv.models import convnext
from acsconv.operators.acsconv import ACSConv
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from torch import Tensor

from utils.utils import update_first_layer


def build_resnet(num_classes=None,
				 in_channels=2,
				 resnet_version=18,
				 pretrained=True) -> models.ResNet:
		resnets = {
			18: models.resnet18, 34: models.resnet34,
			50: models.resnet50, 101: models.resnet101,
			152: models.resnet152
		}
		resnet = resnets[resnet_version](pretrained=pretrained)

		# Update input weights for one input channel
		update_first_layer(resnet, in_channels, True)

		# Modify size of patch embedding vector
		if num_classes is not None:
			linear_size = resnet.fc.in_features
			resnet.fc = torch.nn.Linear(linear_size, num_classes)
		else:
			resnet.fc = torch.nn.Identity() # remove fc head

		resnet = ACSConverter(resnet)

		return resnet.cuda()


def build_convnet(num_classes=None,
				  in_channels=2,
				  convnet_version=18,
				  pretrained=True) -> models.ResNet:
	convnets = {
		'tiny': convnext.convnext_tiny, 'small': convnext.convnext_small,
		'base': convnext.convnext_base, 'large': convnext.convnext_large,
	}
	convnet = convnets[convnet_version](pretrained=pretrained)

	old_pre_layer = convnet.downsample_layers[0][0]
	convnet.downsample_layers[0][0] = ACSConv(
		in_channels,
		old_pre_layer.out_channels,
		kernel_size=old_pre_layer.kernel_size,
		stride=old_pre_layer.kernel_size,
		padding=old_pre_layer.padding,
		groups=old_pre_layer.groups
	)

	# Modify size of patch embedding vector
	if num_classes is not None:
		old_post_layer = convnet.head
		convnet.head = torch.nn.Linear(old_post_layer.in_features, num_classes, bias=True)
	else:
		convnet.head = torch.nn.Identity()  # remove fc head

	return convnet.cuda()
		

class ResNetClassifier(pl.LightningModule):
	def __init__(self, 
				 num_classes,
				 in_channels,
				 net_type,
				 loss_type,
				 version=18,
				 atlas=True,
				 tiles=True,
				 pretrained=True):
		super().__init__()
		self.save_hyperparameters()

		self.net_type = net_type
		self.atlas = atlas
		self.tiles = tiles # activate patch learning

		self.loss_type = loss_type
		if self.loss_type == "TRI":
			self.loss_f = torch.nn.TripletMarginLoss(margin=1.0, swap=True)
		elif self.loss_type == "TRI2":
			self.miner_f = miners.MultiSimilarityMiner()
			self.loss_f = losses.TripletMarginLoss()
			self.loss_h = torch.nn.TripletMarginLoss(margin=1.0, swap=True)
		elif self.loss_type == "SUP":
			self.loss_f = losses.SupConLoss(temperature=0.1)
		elif self.loss_type == "CLA":
			self.loss_f = torch.nn.BCEWithLogitsLoss()
			#input = torch.randn(3, requires_grad=True)
			#target = torch.empty(3).random_(2)
			#output = loss(m(input), target)
			#output.backward()

		# Hooks for intermediate feature map output for visualization
		self.init_features()
		@torch.no_grad()
		def hook_t(module, input, output: Tensor):
			fmap = F.normalize(output, p=2, dim=1)
			self.features_t.append(fmap)

		if net_type == "RESNET":
			self.resnet_model = build_resnet(num_classes=num_classes, in_channels=in_channels, resnet_version=version, pretrained=pretrained)
			self.resnet_model.layer1[-1].register_forward_hook(hook_t)
			self.resnet_model.layer2[-1].register_forward_hook(hook_t)
			self.resnet_model.layer3[-1].register_forward_hook(hook_t)
		elif net_type == "CONVNET":
			self.resnet_model = build_convnet(num_classes=num_classes, in_channels=in_channels, convnet_version=version, pretrained=pretrained)
			self.resnet_model.stages[0][-1].register_forward_hook(hook_t)
			self.resnet_model.stages[1][-1].register_forward_hook(hook_t)
			self.resnet_model.stages[2][-1].register_forward_hook(hook_t)

	def init_features(self):
		self.features_t = []

	def forward(self, X):
		self.init_features()
		return self.resnet_model(X), self.features_t

	def training_step(self, batch, batch_idx):
		x, y, _ = batch

		if self.tiles: # Unsupervised patch learning
			preds_a, _ = self(x[0])
			preds_p, _ = self(x[1])
			preds_n, _ = self(x[2])
			if self.loss_type == 'TRI':
				loss = self.loss_f(preds_a, preds_p, preds_n)
			elif self.loss_type == 'TRI2':
				preds = torch.cat([preds_a, preds_p, preds_n], dim=0)
				labels = torch.tensor([[0] * len(x[0]) + [1] * len(x[0]) + [2] * len(x[0])]).flatten()
				miner_output = self.miner_f(preds, labels)  # in your training for-loop
				loss = self.loss_f(preds, labels, miner_output)
			elif self.loss_type == 'SUP':
				preds = torch.cat([preds_a, preds_p, preds_n], dim=0)
				labels = torch.tensor([[0] * len(x[0]) + [1] * len(x[0]) + [2] * len(x[0])]).flatten()
				loss = self.loss_f(preds, labels)
		else:
			y = F.one_hot(y, num_classes=2)
			loss = self.loss_f(self(x)[0], y.float())

		# Metrics
		batch_size = len(y)
		self.log("train_loss", loss.detach(), batch_size=batch_size)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y, _ = batch

		if self.tiles: # Unsupervised patch learning
			preds_a, fmaps_a = self(x[0])
			preds_p, fmaps_p = self(x[1])
			preds_n, fmaps_n = self(x[2])
			if self.loss_type == 'TRI':
				loss = self.loss_f(preds_a, preds_p, preds_n)
			elif self.loss_type == 'TRI2':
				loss = self.loss_h(preds_a, preds_p, preds_n)
			elif self.loss_type == 'SUP':
				preds = torch.cat([preds_a, preds_p, preds_n], dim=0)
				loss = self.loss_f(preds, torch.tensor([0, 1, 2]))
		else:
			y = F.one_hot(y, num_classes=2)
			loss = self.loss_f(self(x)[0], y.float())

		# Metrics
		batch_size = len(y)
		self.log("val_loss", loss.detach(), batch_size=batch_size)
		self.log("hp_metric", loss.detach(), batch_size=batch_size) # for TensorBoard visualization

		# Visualize feature maps
		# num_tot_vis_imgs = 5
		# num_per_batch = max(round(num_tot_vis_imgs / sum(self.trainer.num_val_batches)), 1)
		# num_prv_imgs = batch_idx * num_per_batch
		# num_vis_imgs = min(num_per_batch, num_tot_vis_imgs - num_prv_imgs)
		# if not self.current_epoch % 2 and num_vis_imgs > 0:
		# 	for i in range(num_vis_imgs):
		# 		self.save_feature_maps([[m[i].cpu().numpy() for m in f] for f in (fmaps_a, fmaps_p, fmaps_n)],
		# 							   [k[i][0, :, :, :].squeeze().cpu().numpy() for k in x],
		# 							   num_prv_imgs + i)

	def predict_step(self, batch: List[Tensor], batch_idx: int):
		x, label, mask = batch

		# in_size = x.shape[2:]
		# # Patch-wise image prediction
		# size = 8
		# stride = 2
		# x = x.unfold(2, size, stride).unfold(3, size, stride)
		# x = x.permute(0, 2, 3, 1, 4, 5)
		# preds, _ = self(x.flatten(end_dim=2))
		# preds = preds.reshape(x.shape[:3] + preds.shape[-1:])
		# preds = preds.permute(0, 3, 1, 2)
		#
		# preds = F.interpolate(preds, size=in_size, mode='bilinear', align_corners=False)
		# return preds
		# x = x[0]
		_, fmaps = self(x)
		out_size = x.shape[-3:]
		for i in range(len(fmaps)):
			# fmap = torch.linalg.vector_norm(fmaps[i], dim=1, keepdim=True)
			fmap = torch.sum(fmaps[i], dim=1, keepdim=True)
			fmap = F.interpolate(fmap, size=out_size, mode='trilinear', align_corners=False)
			fmaps[i] = fmap

		fmaps = torch.cat(fmaps, dim=1)
		return fmaps

	def configure_optimizers(self):
		if self.net_type == 'CONVNET':
			optimizer = torch.optim.AdamW(self.resnet_model.parameters(), lr=5e-5, weight_decay=1e-8)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
			return [optimizer], [scheduler]
		elif self.net_type == 'RESNET':
			optimizer = torch.optim.SGD(self.resnet_model.parameters(), lr=0.1, weight_decay=0.0001)
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
			return [optimizer], [scheduler]

	def save_feature_maps(self,
						  feature_maps: List[List[np.ndarray]],
						  input_imgs: List[np.ndarray],
						  idx: int):
		if self.trainer.log_dir is None: return
		
		fig, axes = plt.subplots(3, 12, figsize=(10, 3))
		for ax in axes.ravel():
			ax.set_axis_off()
		
		for i, (axs, im, fmaps) in enumerate(zip(axes, input_imgs, feature_maps)):
			f64, f32, f16 = (f.sum(axis=0) for f in fmaps)

			axs[0].imshow(im[im.shape[0] // 2, :, :], cmap="gray")
			axs[1].imshow(im[:, im.shape[0] // 2, :], cmap="gray")
			axs[2].imshow(im[:, :, im.shape[0] // 2], cmap="gray")

			axs[3].imshow(f64[f64.shape[0] // 2, :, :], cmap='jet')
			axs[4].imshow(f64[:, f64.shape[0] // 2, :], cmap='jet')
			axs[5].imshow(f64[:, :, f64.shape[0] // 2], cmap='jet')

			axs[6].imshow(f32[f32.shape[0] // 2, :, :], cmap='jet')
			axs[7].imshow(f32[:, f32.shape[0] // 2, :], cmap='jet')
			axs[8].imshow(f32[:, :, f32.shape[0] // 2], cmap='jet')

			axs[9].imshow(f16[f16.shape[0] // 2, :, :], cmap='jet')
			axs[10].imshow(f16[:, f16.shape[0] // 2, :], cmap='jet')
			axs[11].imshow(f16[:, :, f16.shape[0] // 2], cmap='jet')

			# Horizontal titles
			if i == 0:
				axs[0].set_title('Input image (sag)', fontsize=5)
				axs[1].set_title('Input image (cor)', fontsize=5)
				axs[2].set_title('Input image (axi)', fontsize=5)
				axs[3].set_title('Feature map large (sag)', fontsize=5)
				axs[4].set_title('Feature map large (cor)', fontsize=5)
				axs[5].set_title('Feature map large (axi)', fontsize=5)
				axs[6].set_title('Feature map medium (sag)', fontsize=5)
				axs[7].set_title('Feature map medium (cor)', fontsize=5)
				axs[8].set_title('Feature map medium (axi)', fontsize=5)
				axs[9].set_title('Feature map small (sag)', fontsize=5)
				axs[10].set_title('Feature map small (cor)', fontsize=5)
				axs[11].set_title('Feature map small (axi)', fontsize=5)


			# Vertical titles
			axs[0].set_ylabel(("Anchor", "Positive", "Negative")[i], 
				rotation=90)
			axs[0].set_axis_on()
			axs[0].spines[:].set_visible(False)
			axs[0].get_xaxis().set_ticks([])
			axs[0].get_yaxis().set_ticks([])

		fig.tight_layout()
		save_path = os.path.join(self.trainer.log_dir, f"{self.current_epoch}_{idx}.png")
		fig.savefig(save_path, bbox_inches="tight")
		plt.close(fig)

	# def test_step(self, batch, batch_idx):
	# 	x, _, y, _ = batch

	# 	if self.tiles:
	# 		preds_p = self(x[1])
	# 		preds_n = self(x[2])
	# 		preds_a = self(x[0])
	# 		loss = self.loss_f(preds_a, preds_p, preds_n)
	# 	else:
	# 		y = F.one_hot(y, num_classes=2)
	# 		loss = self.loss_f(self(x), y.type(torch.float32))

	# 	# perform logging
	# 	batch_size = len(y)
	# 	self.log("test_loss", loss, batch_size=batch_size)#, on_step=False, on_epoch=True, prog_bar=True, logger=True)

	# def training_epoch_end(self, outputs):
	#     avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
	#     self.log_dict({"avg_train_loss": avg_train_loss})#, "step": self.current_epoch})
