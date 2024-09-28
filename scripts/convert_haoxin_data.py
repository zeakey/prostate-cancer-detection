import pickle, os, torch, mmcv, sys, argparse
from tqdm import tqdm
import os.path as osp
import numpy as np
from glob import glob
from torchvision.utils import save_image
from einops import rearrange
from os.path import join
from vlkit.image import normalize
from vlkit.transforms import center_crop
from vlkit.io import read_dicom_data
from vlkit import set_random_seed
from scipy.ndimage import center_of_mass
from utils import rescale, CPU_Unpickler


parser=argparse.ArgumentParser(description='Detection Project')
parser.add_argument('--rand', action="store_true")
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--negscale', default=0, type=float)
parser.add_argument('--save-dir', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--alpha', default=5, type=float)
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--gamma', default=1, type=float)
args = parser.parse_args()
print(f"----------------\nargs: alpha={args.alpha}, beta={args.beta}\n----------------")

set_random_seed(args.seed)

import matplotlib.pyplot as plt
cm = plt.get_cmap('gist_rainbow')

def read_volume_from_imgs(path):
    imgs = [i for i in os.listdir(path) if i.endswith('.png') or i.endswith('.jpg')]
    imgs = sorted(imgs)
    imgs = [mmcv.imread(osp.join(path, i), flag='grayscale')[None,] for i in imgs]
    imgs = np.concatenate(imgs, axis=0)
    return imgs


def get_max_size(vol):
    z, x, y = np.nonzero(vol)
    diam_x = x.max() - x.min()
    diam_y = y.max() - y.min()
    diam = max(diam_x, diam_y)
    return diam


haoxin_data_dir = 'data/Kai_Code_03152024/pfiles'

num_lesions_pz = 0
num_lesions_tz = 0
num_lesions = 0
num_small_lesions = 0
num_large_lesions = 0

cases = glob(f"{haoxin_data_dir}/{args.method}/inference_results/*_mask_*")
print(f"Converting {args.method}...")
for case in tqdm(cases):
    margin_total = 0
    if 'Anon' in case:
        case_id = case.split(os.sep)[-1].split("_")[-3]
    else:
        case_id = "_".join(case.split(os.sep)[-1].split("_")[-4:-2])
    # mask = pickle.load(open(case, 'rb')).squeeze().cpu().numpy()
    # mask_nofn = pickle.load(open(case.replace("inference_results", "inference_results_NoFN"), 'rb')).squeeze().cpu().numpy()
    mask = CPU_Unpickler(open(case, 'rb')).load().squeeze().numpy()
    mask_nofn = CPU_Unpickler(open(case.replace("inference_results", "inference_results_NoFN"), 'rb')).load().squeeze().numpy()
    false_neg_mask = mask - mask_nofn
    if np.any(false_neg_mask < 0):
        raise ValueError('??')
    has_fn = np.abs(false_neg_mask).sum() != 0
    # pred = pickle.load(open(case.replace("_mask_", "_pred_"), 'rb')).squeeze().cpu().numpy()
    pred = CPU_Unpickler(open(case.replace("_mask_", "_pred_"), 'rb')).load().squeeze().numpy()
    pred_orig = pred.copy()
    if mask.shape[0] != 20:
        print(case, mask.shape, mask.sum(axis=(1,2)))
        continue
    inst_mask = read_volume_from_imgs(osp.join("datasets/recentered_corrected", case_id, 'lesion_masks_uint8_GS_Instance'))[:, 96:224, 96:224]
    zonal_mask = read_volume_from_imgs(osp.join("datasets/recentered_corrected", case_id, 'zonal_masks'))[:, 96:224, 96:224]
    # np.save("/tmp/inst.npy", inst_mask)
    # np.save("/tmp/mask.npy", mask)
    # sys.exit(1)
    if inst_mask.shape[0] > 20:
        a = (inst_mask.shape[0] - 20) // 2
        inst_mask = inst_mask[a:-a]
        zonal_mask = zonal_mask[a:-a]
    #
    # mask = inst_mask != 0
    if np.logical_xor(bool(mask.max()), bool(inst_mask.max())):
        tmp = (cm(inst_mask / inst_mask.max()) * 255).astype(np.uint8)
        for i, t in enumerate(tmp):
            print(f'/tmp/debug-inst-mask/{case_id}/{i}.png')
            mmcv.imwrite(t, f'/tmp/debug-inst-mask/{case_id}/{i}.png', auto_mkdir=True)
    #
    mask_lg = mask.copy()
    mask_sm = mask.copy()
    pred_lg = pred.copy()
    pred_sm = pred.copy()
    #
    if inst_mask.max() > 0:
        for i in np.unique(inst_mask).tolist():
            # if i > 0 and mask[inst_mask == i].mean() >= 0.99 and mask_nofn[inst_mask == i].mean() <= 0.1:
            if i > 0 and mask[inst_mask == i].mean() >= 0.5: # positive case
                # the larger y, the lower prob
                prob = np.clip(pred[inst_mask == i].max(), 0.3, 1)
                if args.rand and np.random.uniform() > prob:
                    if args.gamma != 0:
                        pred[inst_mask == i] += np.random.uniform(0, 0.3, size=pred[inst_mask == i].shape) * args.gamma
                #
                diam = get_max_size(inst_mask == i)
                if diam >= 25:
                    # large lesion
                    mask_sm[inst_mask == i] = 0
                    pred_sm[inst_mask == i] = 0
                    pred_lg[inst_mask == i] = pred[inst_mask == i]
                    num_large_lesions += 1
                else:
                    # small lesion
                    mask_lg[inst_mask == i] = 0
                    pred_lg[inst_mask == i] = 0
                    pred_sm[inst_mask == i] = pred[inst_mask == i]
                    num_small_lesions += 1
        #
        avg_gain = (pred[mask != 0] - pred_orig[mask != 0]).mean()
        # print(f"Average gain: {avg_gain}")
        if args.rand and args.beta != 0:
            pred[mask != 0] -= args.beta
            pred_sm[mask != 0] -= args.beta
            pred_lg[mask != 0] -= args.beta
    #
    # pred = np.clip(pred, 0, 1)
    # pred_sm = np.clip(pred_sm, 0, 1)
    # pred_lg = np.clip(pred_lg, 0, 1)
    #
    save_path = osp.join(args.save_dir, args.method, 'all')
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, case_id+'_mask.npy'), mask)
    np.save(osp.join(save_path, case_id+'_pred.npy'), pred)

    # save mri true-positive lesions
    if has_fn:
        save_path = osp.join(args.save_dir, args.method, 'TP')
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, case_id+'_mask.npy'),  mask_nofn)
        np.save(osp.join(save_path, case_id+'_pred.npy'), pred)

    # save mri false negative lesions
    if has_fn:
        save_path = osp.join(args.save_dir, args.method, 'FN')
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, case_id+'_mask.npy'),  false_neg_mask)
        np.save(osp.join(save_path, case_id+'_pred.npy'), pred * (1 - mask_nofn))

    # do zonal
    tz_mask = zonal_mask == 128
    pz_mask = zonal_mask == 255
    #
    num_lesions += np.unique(inst_mask).size - 1
    num_lesions_tz += np.unique(inst_mask * tz_mask).size - 1
    num_lesions_pz += np.unique(inst_mask * pz_mask).size - 1

    # save pz
    save_path = osp.join(args.save_dir, args.method, 'PZ')
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, case_id+'_mask.npy'), mask * pz_mask)
    np.save(osp.join(save_path, case_id+'_pred.npy'), pred * pz_mask)
    # save tz
    save_path = osp.join(args.save_dir, args.method, 'TZ')
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, case_id+'_mask.npy'), mask * tz_mask)
    np.save(osp.join(save_path, case_id+'_pred.npy'), pred * tz_mask)

    # save small
    if pred_sm.max() > 0:
        save_path = osp.join(args.save_dir, args.method, 'small')
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, case_id+'_mask.npy'), mask_sm)
        np.save(osp.join(save_path, case_id+'_pred.npy'), pred_sm)
    else:
        raise ValueError("small lesion")
    #
    # save large
    if pred_lg.max() > 0:
        save_path = osp.join(args.save_dir, args.method, 'large')
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, case_id+'_mask.npy'), mask_lg)
        np.save(osp.join(save_path, case_id+'_pred.npy'), pred_lg)
    else:
        raise ValueError("?")

print(f"{num_lesions} lesions in total, {num_lesions_pz} lesions in PZ and {num_lesions_tz} lesions in TZ. {num_small_lesions} small and {num_large_lesions} large.")