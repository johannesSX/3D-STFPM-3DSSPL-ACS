import json
import numpy as np
import torch
from dataset.dataset_resnet import MRIDatasetResnet
from dataset.dataset_st import MRIDatasetST


def build_eval(path_to_json):
    with open(path_to_json.format("/data_order/val_brats_eval.json"), 'r') as fin:
        val_brats_eval = json.load(fin)
    with open(path_to_json.format("/data_order/test_brats_eval.json"), 'r') as fin:
        test_brats_eval = json.load(fin)
    with open(path_to_json.format("/data_order/val_ixi_eval.json"), 'r') as fin:
        val_ixi_eval = json.load(fin)
    with open(path_to_json.format("/data_order/test_ixi_eval.json"), 'r') as fin:
        test_ixi_eval = json.load(fin)
    return val_brats_eval, test_brats_eval, val_ixi_eval, test_ixi_eval


def build_dataset(out_img_size, batch_size, n_data_workers, path_to_json, atlas, resample_fac, seq=None, seq_t1c=None, RESNET=False, STPM=False, EVAL=False):
    if RESNET:
        # mean_hist_img = np.load(path_to_json.format("/data_order/histogram_landmarks_t2_BRATS.npy"))
        mean_hist_img = np.load(path_to_json.format("/data_order/histogram_landmarks_t2_BRATS.npy"))
        with open(path_to_json.format("/data_order/train_teacher.json"), 'r') as fin: # train_teacher
            train = json.load(fin)
        with open(path_to_json.format("/data_order/val_teacher.json"), 'r') as fin: # val_teacher
            val = json.load(fin)
        with open(path_to_json.format("/data_order/test_teacher.json"), 'r') as fin: # test_teacher
            test = json.load(fin)
        #train = train[:10]
        train_dataset_gen = MRIDatasetResnet(
            train,
            mean_hist_img,
            out_img_size,
            training=True,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Teacher Train: {}".format(len(train_dataset_gen)))

        #val = val[:10]
        val_dataset_gen = MRIDatasetResnet(
            val,
            mean_hist_img,
            out_img_size,
            training=False,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Teacher Val: {}".format(len(val_dataset_gen)))

        test_dataset_gen = MRIDatasetResnet(
            test,
            mean_hist_img,
            out_img_size,
            training=False,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Teacher Test: {}".format(len(test_dataset_gen)))

    elif STPM:
        mean_hist_img = np.load(path_to_json.format("/data_order/histogram_landmarks_t2_BRATS.npy"))
        with open(path_to_json.format("/data_order/train_student.json"), 'r') as fin:
            train = json.load(fin)
        with open(path_to_json.format("/data_order/val_teacher.json"), 'r') as fin:
            val = json.load(fin)
        with open(path_to_json.format("/data_order/test_teacher.json"), 'r') as fin:
            test = json.load(fin)

        #train = train[:20]

        train_dataset_gen = MRIDatasetST(
            train,
            mean_hist_img,
            out_img_size,
            training=True,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Train: {}".format(len(train_dataset_gen)))

        #val = val[:10] + val[len(val)-10:]
        val_dataset_gen = MRIDatasetST(
            val,
            mean_hist_img,
            out_img_size,
            training=False,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Val: {}".format(len(val_dataset_gen)))

        test_dataset_gen = MRIDatasetST(
            test,
            mean_hist_img,
            out_img_size,
            training=False,
            atlas=atlas,
            resample_fac=resample_fac
        )
        print("Length Test: {}".format(len(test_dataset_gen)))

    train_dataset = torch.utils.data.DataLoader(
        train_dataset_gen, shuffle=True, pin_memory=False,
        batch_size=batch_size,
        num_workers=n_data_workers
    )
    val_dataset = torch.utils.data.DataLoader(
        val_dataset_gen, shuffle=False, pin_memory=False,
        batch_size=1,
        num_workers=n_data_workers
    )
    if test_dataset_gen is not None:
        test_dataset = torch.utils.data.DataLoader(
            test_dataset_gen, shuffle=False, pin_memory=False,
            batch_size=1,
            num_workers=n_data_workers
        )
    else:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset