import copy

import numpy as np
import random
import torch
import torchio as tio
import tqdm

from sklearn import \
    utils as sk_utils

from dataset.superdataset import SuperDataset


class MRIDatasetST(torch.utils.data.Dataset, SuperDataset):

    def transforms_3D(self, img):
        landmarks_dict = {'mri': self.mean_hist_img}
        hist_transform = tio.HistogramStandardization(landmarks_dict)
        z_transform = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        crop_transform = tio.CropOrPad(self.out_size)
        rescale_transform = tio.RescaleIntensity(
            out_min_max=(0, 1)
        )
        resample_transform = tio.Resample(self.resample_fac) # 2.75
        # affine_transform = tio.RandomAffine(
        #     scales=(1.0, 1.2),
        #     degrees=(-10, 10),
        #     isotropic=True,
        # )
        # flip_transform = tio.RandomFlip()

        if self.training:
            transforms = tio.Compose([hist_transform, z_transform, crop_transform, resample_transform, rescale_transform])
        else:
            transforms = tio.Compose([hist_transform, z_transform, crop_transform, resample_transform, rescale_transform])
        return transforms(img)


    def __init__(self, pt_dataset, mean_hist_img, out_size, use_data_aug=False, resample_fac=2.75, training=False, atlas=False, seed=0):
        np.random.seed(seed)
        random.seed(seed)

        self.mean_hist_img = mean_hist_img
        self.out_size = out_size
        self.resample_fac = resample_fac
        self.training = training
        self.atlas = atlas

        pt_dataset = sk_utils.shuffle(pt_dataset, random_state=41)
        label = 't2'
        _pt_dataset = []
        for data_dict in tqdm.tqdm(pt_dataset):
            if label in data_dict and len(data_dict[label]) != 0:
                _pt_dataset.append(data_dict)
        self.pt_dataset = _pt_dataset

        file_atlas = "./data/masks/mni_icbm152_nlin_sym_09c_CerebrA_nifti/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii"
        mri_atlas_dd = tio.RescaleIntensity(
            out_min_max=(0, 1)
        )(tio.ScalarImage(file_atlas)).data

        mri_atlas = tio.LabelMap(file_atlas)
        mri_atlas.data = mri_atlas_dd
        self.mri_atlas = mri_atlas


    def __len__(self):
        return len(self.pt_dataset)


    def get_helper(self, idx):
        return self.__getitem__(idx)


    def __getitem__(self, idx):
        data_dict = self.pt_dataset[idx]
        if 't2' in data_dict and len(data_dict['t2']) != 0:
            file = data_dict['t2'][0]
        seg = data_dict["seg"]

        mri_img = tio.ScalarImage(file)
        if len(seg) > 0:
            mri_seg = tio.LabelMap(seg, affine=mri_img.affine)
        else:
            mri_seg = torch.zeros((1, mri_img.data.shape[1], mri_img.data.shape[2], mri_img.data.shape[3])).type(torch.float)
            mri_seg = tio.LabelMap(
                tensor=mri_seg,
                affine=mri_img.affine,
                direction=mri_img.direction,
                orientation=mri_img.orientation,
                origin=mri_img.origin,
                spacing=mri_img.spacing
            )

        if data_dict['keyword'][0] != 'physionet':
            label_not_healthy = int(data_dict["not_healthy"][0])
        else:
            if mri_seg.data.max() == 0:
                label_not_healthy = 0
            elif  mri_seg.data.max() > 0:
                label_not_healthy = 1

        mri_sub = tio.Subject(
            mri=mri_img,
            atlas=copy.deepcopy(self.mri_atlas),
            seg=mri_seg
        )
        mri_sub = self.transforms_3D(mri_sub)
        mri_sub.seg.data = mri_sub.seg.data.round().clip(max=1).type(torch.int)
        # mri_sub.seg.data[mri_sub.seg.data != 1] = 0
        # mri_sub.seg.data = mri_sub.seg.data.type(torch.int)

        if self.atlas:
            mri_img = torch.cat([mri_sub.mri.data, mri_sub.atlas.data], dim=0)
        else:
            mri_img = mri_sub.mri.data

        return [mri_img, label_not_healthy, mri_sub.seg.data]
