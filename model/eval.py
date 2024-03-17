import os
import pathlib

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as tom

from sklearn import model_selection as sk_mselection
from sklearn import cluster as sk_cluster
from model.stpm import STPM
from run_stpm3D import config_hyperparameters
from tqdm import tqdm
from typing import List

from dataset.dataset_st import MRIDatasetST



def save_anomaly_map(anomaly_map: np.ndarray, amaps: List[np.ndarray], input_img: np.ndarray, mask: np.ndarray, idx: int, log_dir: str):

    am64, am32, am16 = amaps

    def f_slice(i, img, type="sag"):
        if type == "sag":
            img = img[i, :, :]
        elif type == "cor":
            img = img[:, i, :]
        elif type == "axi":
            img = img[:, :, i]
        return img

    fig, axes = plt.subplots(3, 7, figsize=(20, 6))
    fig.tight_layout()
    for ax in axes.ravel():
        ax.set_axis_off()

    ims = []
    for frame_i in range(input_img.shape[0]):

        sub_ims = []
        taxes = ["sag", "cor", "axi"]
        for i, (axs, type) in enumerate(zip(axes, taxes)):
            animated = True
            if frame_i == 0:
                animated = False

            axs[0].set_title('Input image ({})'.format(type), fontsize=10)
            sub_ims.append(axs[0].imshow(f_slice(frame_i, input_img, type), cmap='gray', animated=animated))

            axs[1].set_title('Anomaly map large', fontsize=10)
            sub_ims.append(axs[1].imshow(f_slice(frame_i, am64, type), animated=animated))

            axs[2].set_title('Anomaly map medium', fontsize=10)
            sub_ims.append(axs[2].imshow(f_slice(frame_i, am32, type), animated=animated))

            axs[3].set_title('Anomaly map small', fontsize=10)
            sub_ims.append(axs[3].imshow(f_slice(frame_i, am16, type), animated=animated))

            axs[4].set_title('Anomaly map', fontsize=10)
            sub_ims.append(axs[4].imshow(f_slice(frame_i, anomaly_map, type), animated=animated))

            axs[5].set_title('Anomaly map on image', fontsize=10)
            sub_ims.append(axs[5].imshow(f_slice(frame_i, input_img, type), cmap="gray", animated=animated))
            sub_ims.append(
                axs[5].imshow(f_slice(frame_i, anomaly_map, type), alpha=0.5, animated=animated))

            sub_ims.append(axs[6].set_title('Ground truth mask', fontsize=10))
            sub_ims.append(axs[6].imshow(f_slice(frame_i, mask, type), cmap="gray", animated=animated))

        if frame_i % 10 == 0:
            save_path = os.path.join(log_dir, f"fmaps_{idx}_{frame_i}.png")
            plt.savefig(save_path)

        ims.append(sub_ims)

    fig.tight_layout()
    anim = animation.ArtistAnimation(fig, ims, interval=2, blit=True, repeat_delay=10)
    save_path = os.path.join(log_dir, f"fmaps_{idx}.mp4")
    writervideo = animation.FFMpegWriter(fps=10)
    anim.save(save_path, writer=writervideo)
    # anim.save(save_path, writer='imagemagick')
    plt.close(fig)
    plt.close('all')


def save_roc(fpr: np.ndarray,
             tpr: np.ndarray,
             threshs: np.ndarray,
             log_dir: str,
             fold_nr: int):
    # Optimal threshold
    opt_idx = np.argmax(tpr - fpr)

    fig, ax = plt.subplots(figsize=(4, 4))

    # ax.plot(fpr, tpr, label=f"STPM ({auroc:.2f})")
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--k", label="Chance (0.50)")

    step = max(round(len(threshs) / 10), 1)
    for x, y, t in zip(fpr[::step], tpr[::step], threshs[::step]):
        ax.plot([x], [y], "ok", fillstyle="none")
        ax.annotate(f"{t:.2g}", (x + 0.01, y - 0.05))
    ax.plot(fpr[[opt_idx]], tpr[[opt_idx]], 'o', fillstyle="none",
            label=f"Optimal ({threshs[opt_idx]:.2g})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    save_path = os.path.join(log_dir, f"{fold_nr}_roc.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    plt.close('all')


def calc_kmeans(anomaly_map, threshold=None):
    anomaly_map_np = anomaly_map.cpu().numpy()
    if threshold is None:
        tempimg_th = np.percentile(anomaly_map_np, 99)
    else:
        tempimg_th = threshold
    tempimg = np.float32(anomaly_map_np >= tempimg_th)
    temploc = np.reshape(tempimg, (-1, 1))

    kmeans = sk_cluster.KMeans(n_clusters=2, random_state=0).fit(
        temploc
    )
    labels = kmeans.predict(temploc)
    label_img = labels.reshape(anomaly_map_np.shape)

    label_img = torch.from_numpy(label_img).to(anomaly_map.dtype)
    assert len(torch.unique(label_img)) == 2
    return label_img


def draw_fmaps(model, dataset, save_path, KMEANS=True):
    for idx, batch in enumerate(tqdm(dataset)):
        x, label, mask = batch
        with torch.no_grad():
            # Inference
            features_t, features_s = model.forward(x)

        anomaly_map, amaps = model.anomaly_map(features_t, features_s, x.shape[-3:])

        if KMEANS:
            anomaly_map = calc_kmeans(anomaly_map)

        save_anomaly_map(
            anomaly_map.squeeze().cpu().numpy(), [m.squeeze().cpu().numpy() for m in amaps],
            x[:, 0, :, :, :].squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(), idx, save_path
        )
        # save_anomaly_map_slice(
        #     anomaly_map.squeeze().cpu().numpy(), [m.squeeze().cpu().numpy() for m in amaps],
        #     x[:, 0, :, :, :].squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(), idx, 220, save_path
        # )
        # if KMEANS:
        #     anomaly_map = anomaly_map.cpu().numpy()
        #     tempimg_th = np.percentile(anomaly_map, 99)
        #     tempimg = np.float32(anomaly_map >= tempimg_th)
        #     temploc = np.reshape(tempimg, (-1, 1))
        #
        #     kmeans = sk_cluster.KMeans(n_clusters=2, random_state=0).fit(
        #         temploc
        #     )
        #     labels = kmeans.predict(temploc)
        #     label_img = labels.reshape(anomaly_map.shape)
        #
        #     label_img = nib.Nifti1Image(label_img.squeeze(), affine=np.eye(4))
        #     nib.save(label_img, '../results/2022_09_12_STPM3D/miao_1.nii.gz')



def calc_cla_metrics(model, dataset, KMEANS=True):
    ovloss = []
    ovlabel = []
    ovscore = []
    for batch in tqdm(dataset):
        x, label, mask = batch
        with torch.no_grad():
            features_t, features_s = model.forward(x)
        loss = model.loss_function(features_t, features_s)

        ovloss.append(loss.cpu())

        anomaly_map, amaps = model.anomaly_map(features_t, features_s, x.shape[-3:])
        if KMEANS:
            anomaly_map = calc_kmeans(anomaly_map)
        score = anomaly_map.amax((1, 2, 3, 4))

        anomaly_score = score + 1.0 * loss

        ovlabel.append(label.cpu())
        ovscore.append(torch.tensor([anomaly_score.cpu()]))
        # ovscore.append(score.cpu())

    ovloss = torch.stack(ovloss)
    ovlabel = torch.stack(ovlabel)
    ovscore = torch.stack(ovscore)

    rocaucscore = tom.functional.auroc(ovscore, ovlabel, task='binary')
    avprecscore = tom.functional.average_precision(ovscore, ovlabel, task='binary')
    fpr, tpr, threshs = tom.functional.roc(ovscore, ovlabel, task='binary')

    return torch.mean(ovloss), rocaucscore, avprecscore, fpr, tpr, threshs

def calc_seg_metrics(model, dataset, threshold, save_path, KMEANS=True):
    ovrocaucscore = []
    oviouscore = []
    ovdicescore = []
    ovspecscore = []
    ovprecscore = []
    ovrecscore = []
    for batch in tqdm(dataset):
        x, label, mask = batch
        with torch.no_grad():
            # Inference
            features_t, features_s = model.forward(x)
        anomaly_map, amaps = model.anomaly_map(features_t, features_s, x.shape[-3:])

        if KMEANS:
            anomaly_map = calc_kmeans(anomaly_map)
        # if mask.flatten().cpu().max() >= 1:
        #     ovidx.append(True)
        # else:
        #     ovidx.append(False)

        mask = mask.flatten().cpu().reshape(-1, 1)
        map = anomaly_map.flatten().cpu().reshape(-1, 1)

        rocaucscore = tom.functional.auroc(map.flatten(), mask.flatten(), task='binary')
        iouscore = tom.functional.jaccard_index(map, mask, num_classes=2, threshold=threshold, task='binary')
        dicescore = tom.functional.dice(map, mask, threshold=threshold)
        specscore = tom.functional.specificity(map, mask, threshold=threshold, task='binary')
        # aucscore = tom.functional.auc(map.flatten(), mask.flatten(), reorder=True, task='binary')
        precscore, recscore, _ = tom.functional.precision_recall_curve(map, mask, thresholds=torch.as_tensor([threshold]), task='binary')

        ovrocaucscore.append(rocaucscore)
        oviouscore.append(iouscore)
        ovdicescore.append(dicescore)
        ovspecscore.append(specscore)
        # ovaucscore.append(aucscore)
        ovprecscore.append(precscore)
        ovrecscore.append(recscore)

    #sensitivity_list, FPavg_list, _ = computeFROC(ovmap.numpy(), ovmask.numpy().astype(int), allowedDistance=2)
    #plotFROC(sensitivity_list, FPavg_list, os.path.join(save_path, f"froc.png"))

    return torch.mean(torch.stack(ovrocaucscore)), torch.mean(torch.stack(oviouscore)), \
           torch.mean(torch.stack(ovdicescore)), torch.mean(torch.stack(ovspecscore)), \
           torch.mean(torch.stack(ovprecscore)), torch.mean(torch.stack(ovrecscore))


def calc_seg_threshold(model, dataset, KMEANS, save_path, len_of_splits=50):
    ovthreshs = []
    for batch in tqdm(dataset):
        x, _, mask = batch
        with torch.no_grad():
            # Inference
            features_t, features_s = model.forward(x)

        anomaly_map, amaps = model.anomaly_map(features_t, features_s, x.shape[-3:])

        # if KMEANS:
        #     anomaly_map = calc_kmeans(anomaly_map)

        mask = mask.flatten().cpu().reshape(1, -1)
        map = anomaly_map.flatten().cpu().reshape(1, -1)
        fpr, tpr, threshs = tom.functional.roc(map, mask, task='binary')
        threshold = threshs[torch.argmax(tpr - fpr)].item()
        ovthreshs.append(threshold)
        # save_roc(fpr.numpy(), tpr.numpy(), threshs.numpy(), save_path, kf_idx)
    return None, None, None, np.mean(ovthreshs)


def calc_seg_threshold_kf(model, dataset, KMEANS, save_path, len_of_splits=50):
    if len(dataset) >= len_of_splits:
        splits = len(dataset) // len_of_splits
    else:
        splits = 2
    kf = sk_mselection.KFold(n_splits=splits, shuffle=False, random_state=None)
    ovthreshs = []
    kf_idx = 0
    for train_icds, test_icds in tqdm(kf.split(np.arange(0, len(dataset), 1, dtype=int)), total=kf.get_n_splits()):
        ovmask = []
        ovmap = []
        for idx in test_icds:
            x, _, mask = dataset.dataset[idx]
            x, mask = torch.unsqueeze(x, dim=0), torch.unsqueeze(mask, dim=0)
            with torch.no_grad():
                # Inference
                features_t, features_s = model.forward(x)

            anomaly_map, amaps = model.anomaly_map(features_t, features_s, x.shape[-3:])

            # if KMEANS:
            #     anomaly_map = calc_kmeans(anomaly_map)

            ovmask.append(mask.flatten().cpu())
            ovmap.append(anomaly_map.flatten().cpu())
        ovmask = torch.stack(ovmask)
        ovmap = torch.stack(ovmap)
        fpr, tpr, threshs = tom.functional.roc(ovmap, ovmask)
        threshold = threshs[torch.argmax(tpr - fpr)].item()
        ovthreshs.append(threshold)
        # save_roc(fpr.numpy(), tpr.numpy(), threshs.numpy(), save_path, kf_idx)
        kf_idx += 1
    return fpr, tpr, threshs, np.mean(ovthreshs)



def run_eval(KMEANS=False, MODE="SEG", version=11198, ckpt='last_epoch_student.ckpt'):
    CKPT_PATH = "./data/lightning_logs/version_{}/checkpoints/{}"

    path = CKPT_PATH.format(version, ckpt)
    args_gen, _, args_stpm = config_hyperparameters()
    pl.seed_everything(42, workers=True)

    mean_hist_img = np.load(args_gen.data_dir.format("/data_order/histogram_landmarks_t1_BRATS.npy"))

    from dataset.dataloader import build_eval
    val_brats_eval, test_brats_eval, val_ixi_eval, test_ixi_eval = build_eval(args_gen.data_dir)
    eval_brats = val_brats_eval
    eval_ixi = val_ixi_eval

    model = STPM.load_from_checkpoint(path)

    save_path = (pathlib.Path(path).parent).parent / MODE
    os.makedirs(save_path, exist_ok=True)

    if MODE == "CLASS":
        loss, rocaucscore, avprecscore, threshs = [], [], [], []
        for idx_s in range(0, len(eval_brats), len(eval_ixi)):
            idx_e = idx_s + len(eval_ixi)
            dataset = MRIDatasetST(
                eval_brats[idx_s: idx_e] + eval_ixi,
                mean_hist_img,
                args_gen.out_img_size,
                training=False,
                atlas=args_gen.atlas,
                resample_fac=args_gen.resample_fac
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, shuffle=False,
                pin_memory=False, batch_size=1, num_workers=0
            )
            _loss, _rocaucscore, _avprecscore, _, _, _threshs = calc_cla_metrics(model.cpu().eval(), dataloader, KMEANS=False)
            loss.append(float(_loss))
            rocaucscore.append(float(_rocaucscore))
            avprecscore.append(float(_avprecscore))
        print(
            "LOSS:      {} \n" \
            "AUROC:     {} \n" \
            "AVGPREC:   {} ".format(np.mean(loss), np.mean(rocaucscore), np.mean(avprecscore))
        )
        d = {'LOSS': loss, 'AUROC': rocaucscore, 'AVGPREC': avprecscore}
        df = pd.DataFrame(data=d)
        df.to_csv(save_path / '{}_{}_{}.csv'.format(MODE, KMEANS, ckpt.replace('.ckpt', '')))

    elif MODE == "SEG":
        # eval_brats = eval_brats[:10]
        dataset = MRIDatasetST(
            eval_brats,
            mean_hist_img,
            args_gen.out_img_size,
            training=False,
            atlas=args_gen.atlas,
            resample_fac=args_gen.resample_fac
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            pin_memory=False, batch_size=1, num_workers=0
        )
        _, _, _, threshold = calc_seg_threshold(model.cpu().eval(), dataloader, KMEANS, save_path)
        print("Threshold: {}".format(threshold))

        rocaucscoreseg, iouscore, dicescore, specscore, precscore, recscore = calc_seg_metrics(
            model.cpu().eval(), dataloader, threshold, save_path, KMEANS
        )
        aucscore = 0.0
        #threshold = 9.306965148425661e-05
        #rocaucscoreseg, iouscore, dicescore, specscore, aucscore, precscore, recscore = calc_seg_metrics(model.cpu().eval(), dataset, threshold, save_path, KMEANS)

        print(
            "THRESHOLD: {} \n" \
            "AUROC:     {} \n" \
            "IOU:       {} \n" \
            "DICE:      {} \n" \
            "SPECI:     {} \n" \
            "SEN:       {} \n" \
            "PREC:      {} \n" \
            "AUC        {} \n".format(
                float(threshold), float(rocaucscoreseg), float(iouscore), float(dicescore), float(specscore), float(recscore), float(precscore), float(aucscore)
            )
        )
        d = {'THRESHOLD': [float(threshold)],
             'AUROC': [float(rocaucscoreseg)],
             'IOU': [float(iouscore)],
             'DICE': [float(dicescore)],
             'SPECI': [float(specscore)],
             'SEN': [float(recscore)],
             'PREC': [float(precscore)],
             'AUC': [float(aucscore)]}
        df = pd.DataFrame(data=d)
        df.to_csv(save_path / '{}_{}_{}.csv'.format(MODE, KMEANS, ckpt.replace('.ckpt', '')))

    elif MODE == "FMAPS":
        dataset = MRIDatasetST(
            eval_brats,
            mean_hist_img,
            args_gen.out_img_size,
            training=False,
            atlas=args_gen.atlas,
            resample_fac=args_gen.resample_fac
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False,
            pin_memory=False, batch_size=1, num_workers=0
        )
        draw_fmaps(model.cpu().eval(), dataloader, save_path, KMEANS)