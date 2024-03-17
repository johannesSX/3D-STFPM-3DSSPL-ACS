import copy

import torch
import torchio as tio

import nibabel as nib
import numpy as np
import random
import tqdm

from sklearn import \
    utils as sk_utils

from dataset.superdataset import SuperDataset


class MRIDatasetResnet(torch.utils.data.Dataset, SuperDataset):

    def transforms_3D(self, img):
        landmarks_dict = {'mri': self.mean_hist_img}
        hist_transform = tio.HistogramStandardization(landmarks_dict)
        z_transform = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        crop_transform = tio.CropOrPad(self.out_size)
        rescale_transform = tio.RescaleIntensity(
            out_min_max=(0, 1),
            # percentiles=(0.5, 99.5)
        )
        resample_transform = tio.Resample(self.resample_fac) # 2.75
        # affine_transform = tio.RandomAffine(
        #     scales=(1.0, 1.2),
        #     degrees=(-10, 10),
        #     isotropic=True,
        #     default_pad_value='minimum',
        # )
        flip_transform = tio.RandomFlip()

        if self.training:
            transforms = tio.Compose([hist_transform, z_transform, crop_transform, resample_transform, rescale_transform])
        else:
            transforms = tio.Compose([hist_transform, z_transform, crop_transform, resample_transform, rescale_transform])
        return transforms(img)


    def create_triplet(self, mri_sub, plot=False):
        c_size, x_size, y_size, z_size = mri_sub.mri.shape

        img_a = torch.zeros((1, 32, 32, 32))
        itr = 0
        while img_a.flatten().max() == img_a.flatten().min():
            x = random.randint(0, x_size - 32 - 1)
            y = random.randint(0, y_size - 32 - 1)
            z = random.randint(0, z_size - 32 - 1)
            img_a = mri_sub.mri.data[:, x : x + 32, y : y + 32, z : z + 32]
            if itr > 25:
                break
            itr += 1
        atl_a = mri_sub.atlas.data[:, x: x + 32, y: y + 32, z: z + 32]

        img_p = torch.zeros((1, 32, 32, 32))
        itr = 0
        while img_p.flatten().max() == img_p.flatten().min():
            if x < 32:
                t_x = np.random.choice([0, 32 - 1])
            elif x > x_size - 32 - 1:
                t_x = np.random.choice([-32, 0])
            else:
                t_x = np.random.choice([-32, 0, 32 - 1])

            if y < 32:
                t_y = np.random.choice([0, 32 - 1])
            elif y > y_size - 32 - 1:
                t_y = np.random.choice([-32, 0])
            else:
                t_y = np.random.choice([-32, 0, 32 - 1])

            if z < 32:
                t_z = np.random.choice([0, 32 - 1])
            elif z > z_size - 32 - 1:
                t_z = np.random.choice([-32, 0])
            else:
                t_z = np.random.choice([-32, 0, 32 - 1])

            # t_x = random.randint(-16, 16)
            # t_y = random.randint(-16, 16)
            # t_z = random.randint(-16, 16)
            # if x + t_x < 0 or x + 32 + t_x >= x_size - 1 \
            #         or y + t_y < 0 or y + 32 + t_y >= y_size - 1 \
            #         or z + t_z < 0 or z + 32 + t_z >= z_size - 1:
            #     img_p = torch.zeros((1, 32, 32, 32))
            if x + t_x < 0 or x + 32 + t_x >= x_size - 1 \
                    or y + t_y < 0 or y + 32 + t_y >= y_size - 1 \
                    or z + t_z < 0 or z + 32 + t_z >= z_size - 1:
                img_p = torch.zeros((1, 32, 32, 32))
            else:
                img_p = mri_sub.mri.data[:, x + t_x : x + 32 + t_x, y + t_y: y + 32 + t_y, z + t_z : z + 32 + t_z]
            if itr > 25:
                break
            itr += 1
        atl_p = mri_sub.atlas.data[:, x + t_x: x + 32 + t_x, y + t_y: y + 32 + t_y, z + t_z: z + 32 + t_z]
        # img_p = tio.transforms.RandomBlur()(img_p)

        data_dict = random.choice(self.pt_dataset)
        for label in ["t1"]:
            if label in data_dict and len(data_dict[label]) > 0:
                file = data_dict[label][0]
                break
        n_mri_img = tio.Subject(
            mri=tio.ScalarImage(file)
        )
        n_mri_img = self.transforms_3D(n_mri_img).mri

        img_n = torch.zeros((1, 32, 32, 32))
        itr = 0
        while img_n.flatten().max() == img_n.flatten().min():
            n_x = random.randint(0, x_size - 32)
            n_y = random.randint(0, y_size - 32)
            n_z = random.randint(0, z_size - 32)
            if n_x in list(range(x, x + 32)) or n_y in list(range(y, y + 32)) or n_z in list(range(z, z + 32)):
                img_n = torch.zeros((1, 32, 32, 32))
            else:
                img_n = n_mri_img.data[:, n_x: n_x + 32, n_y: n_y + 32, n_z: n_z + 32]
            if itr > 25:
                break
            itr += 1
        atl_n = mri_sub.atlas.data[:, n_x: n_x + 32, n_y: n_y + 32, n_z: n_z + 32]
        del data_dict
        #del mri_img

        if plot:
            _mri_img_1 = copy.deepcopy(mri_sub.mri)
            _mri_img_1.data[:, n_x : n_x + 32, n_y : n_y + 32, n_z : n_z + 32] = img_n
            _mri_img_1.data[:, x + t_x: x + 32 + t_x, y + t_y: y + 32 + t_y, z + t_z: z + 32 + t_z] = img_p
            _mri_img_1.data[:, x : x + 32, y : y + 32, z : z + 32] = img_a

            _mri_img_2 = copy.deepcopy(mri_sub.mri)
            _mri_img_2.data[:, n_x: n_x + 32, n_y: n_y + 32, n_z: n_z + 32] = 5
            _mri_img_2.data[:, x + t_x: x + 32 + t_x, y + t_y: y + 32 + t_y, z + t_z: z + 32 + t_z] = 4
            _mri_img_2.data[:, x: x + 32, y: y + 32, z: z + 32] = 3

            _atl_img_1 = copy.deepcopy(mri_sub.atlas)
            _atl_img_1.data[:, n_x: n_x + 32, n_y: n_y + 32, n_z: n_z + 32] = 5
            _atl_img_1.data[:, x + t_x: x + 32 + t_x, y + t_y: y + 32 + t_y, z + t_z: z + 32 + t_z] = 4
            _atl_img_1.data[:, x: x + 32, y: y + 32, z: z + 32] = 3

            _mri_img_2.to_gif(
                axis=2,
                duration=20,
                output_path='../results/2022_08_22_STPM3D/tiles_axi.gif'
            )
            _mri_img_2.to_gif(
                axis=1,
                duration=20,
                output_path='../results/2022_08_22_STPM3D/tiles_cor.gif'
            )
            _mri_img_2.to_gif(
                axis=3,
                duration=20,
                output_path='../results/2022_08_22_STPM3D/tiles_sag.gif'
            )

            _mri_img_1 = nib.Nifti1Image(_mri_img_1.data.squeeze().numpy(), affine=np.eye(4))
            _mri_img_2 = nib.Nifti1Image(_mri_img_2.data.squeeze().numpy(), affine=np.eye(4))
            _atl_img_1 = nib.Nifti1Image(_atl_img_1.data.squeeze().numpy(), affine=np.eye(4))
            nib.save(_mri_img_1, '../results/2022_08_22_STPM3D/miao_1.nii.gz')
            nib.save(_mri_img_2, '../results/2022_08_22_STPM3D/miao_2.nii.gz')
            nib.save(_atl_img_1, '../results/2022_08_22_STPM3D/miao_3.nii.gz')

        # resize_transform = tio.transforms.Resize(
        #     target_shape=[
        #         int(self.out_size[0] / self.resample_fac),
        #         int(self.out_size[1] / self.resample_fac),
        #         int(self.out_size[2] / self.resample_fac)
        #     ]
        # )

        # img_a = [
        #     resize_transform(img_a),
        #     resize_transform(atl_a),
        # ]
        # img_p = [
        #     resize_transform(img_p),
        #     resize_transform(atl_p),
        # ]
        # img_n = [
        #     resize_transform(img_n),
        #     resize_transform(atl_n),
        # ]

        return [[img_a, atl_a], [img_p, atl_p], [img_n, atl_n]]


    def __init__(self, pt_dataset, mean_hist_img, out_size, resample_fac=2.75, training=False, atlas=False, seed=0):
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


    def __getitem__(self, idx):
        data_dict = self.pt_dataset[idx]
        if 't2' in data_dict and len(data_dict['t2']) != 0:
            file = data_dict['t2'][0]
        label_not_healthy = int(data_dict["not_healthy"][0])

        mri_seg = torch.zeros((1,
                               int(self.out_size[0] / self.resample_fac),
                               int(self.out_size[1] / self.resample_fac),
                               int(self.out_size[2] / self.resample_fac)))
        mri_sub = tio.Subject(
            mri=tio.ScalarImage(file),
            atlas=copy.deepcopy(self.mri_atlas)
        )
        mri_sub = self.transforms_3D(mri_sub)
        # mri_sub.mri.save('../results/miao.nii.gz')

        mri_tiles = self.create_triplet(mri_sub)
        if self.atlas:
            for idx, tile in enumerate(mri_tiles):
                mri_tiles[idx] = torch.cat(tile, dim=0)
        else:
            for idx, tile in enumerate(mri_tiles):
                mri_tiles[idx] = tile[0]

        return [mri_tiles, label_not_healthy, mri_seg]
