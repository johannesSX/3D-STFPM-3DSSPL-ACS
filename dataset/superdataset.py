import copy
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import tqdm


class SuperDataset():

    def to_slices(self, img, seg, idx):
        i = (self.tmp_size[2] // 2) + idx
        img = img[:, :, i]
        seg = seg[:, :, i]
        return img, seg

    def to_tiles(self, imgs, idx):
        #plt.imshow(img.squeeze(), cmap="gray")
        #plt.title(idx)
        #plt.show()
        _imgs = []
        for img in imgs:
            x, y = self.tiles_lookup[idx]
            x_s, x_t = x, x + self.tile_size
            y_s, y_t = y, y + self.tile_size

            _img = img[:, y_s : y_t, x_s : x_t]
            _imgs.append(_img)
        #plt.imshow(_img.squeeze(), cmap="gray")
        #plt.title(idx)
        #plt.show()
        #plt.imshow(_seg.squeeze(), cmap="gray")
        #plt.show()
        return _imgs

    def create_tile_lookup(self, tile_size):
        tiles_lookup = []
        for j in range(0, self.tmp_size[0], tile_size):
            for i in range(0, self.tmp_size[0], tile_size):
                tiles_lookup.append([j, i])
        return tiles_lookup

    def to_tiles_dict(self, data_dict):
        lst_data_dict = []
        for idx in self.n_tiles:
            tuple_dict = copy.deepcopy(data_dict)
            tuple_dict["tile"] = idx
            lst_data_dict.append(tuple_dict)
        return lst_data_dict

    def to_slices_dict(self, data_dict):
        lst_data_dict = []
        for idx in self.n_slices:
            tuple_dict = copy.deepcopy(data_dict)
            tuple_dict["idx"] = idx
            lst_data_dict.append(tuple_dict)
        return lst_data_dict

    def data_to_tuple_dict(self, data_dict):
        data_dict = copy.deepcopy(data_dict)
        if len(data_dict["seg"]) == 0:
            seg = None
        else:
            seg = data_dict["seg"][0]
        del data_dict["seg"]

        not_healthy = data_dict["not_healthy"][0]
        del data_dict["not_healthy"]

        del data_dict["mask"]
        del data_dict["class_label"]

        for label, (k, value_lst) in enumerate(data_dict.items()):
            if len(value_lst) != 0:
                tuple_dict = copy.deepcopy(self.tuple_dict)
                tuple_dict["seq"] = value_lst[0]
                tuple_dict["label"] = k
                tuple_dict["seg"] = seg
                tuple_dict["not_healthy"] = not_healthy
        return tuple_dict


    def torch_normalize(self, x, a, b, p_min=None, p_max=None):
        if p_min is None and p_max is None:
            x = (b - a) * ((x - x.min()) / (x.max() - x.min())) + a
        else:
            x = (b - a) * ((x - p_min) / (p_max - p_min)) + a
        return x


    def get_mean_std(self, data_dict):
        file = data_dict["seq"]
        slice_idx = data_dict["idx"]

        mri_img = nib.load(file).get_fdata()
        mri_seg = np.zeros((self.tmp_size[0], self.tmp_size[1], self.tmp_size[2]))
        _mri_img, _ = self.to_slices(mri_img, mri_seg, slice_idx)
        mean = np.mean(_mri_img)
        std = np.std(_mri_img)
        return mean, std