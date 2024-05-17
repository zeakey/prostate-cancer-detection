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


parser=argparse.ArgumentParser(description='Detection Project')
parser.add_argument('--seed',default=0,type=int,help='number of total epochs to run')
parser.add_argument('--save-dir', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--randscale', default=0.1, type=float)
args = parser.parse_args()


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
    diam = min(diam_x, diam_y)
    return diam


haoxin_data_dir = '/media/hdd2/prostate-cancer-with-dce/Kai_Code_03152024/pfiles'


cases = glob(f"{haoxin_data_dir}/{args.method}/inference_results/*_mask_*")
print(f"Converting {args.method}...")
for case in cases:
    if 'Anon' in case:
        case_id = case.split(os.sep)[-1].split("_")[-3]
    else:
        case_id = "_".join(case.split(os.sep)[-1].split("_")[-4:-2])
    mask = pickle.load(open(case, 'rb')).squeeze().cpu().numpy()
    mask_nofn = pickle.load(open(case.replace("inference_results", "inference_results_NoFN"), 'rb')).squeeze().cpu().numpy()
    diff = mask - mask_nofn
    if np.any(diff < 0):
        raise ValueError('??')
    has_fn = np.abs(diff).sum() != 0
    pred = pickle.load(open(case.replace("_mask_", "_pred_"), 'rb')).squeeze().cpu().numpy()
    if mask.shape[0] != 20:
        print(case, mask.shape, mask.sum(axis=(1,2)))
        continue
    inst_mask = read_volume_from_imgs(osp.join("datasets/recentered_corrected", case_id, 'lesion_masks_uint8_GS_Instance'))[:, 96:224, 96:224]
    if inst_mask.shape[0] > 20:
        a = (inst_mask.shape[0] - 20) // 2
        inst_mask = inst_mask[a:-a]
    if np.logical_xor(bool(mask.max()), bool(inst_mask.max())):
        tmp = (cm(inst_mask / inst_mask.max()) * 255).astype(np.uint8)
        for i, t in enumerate(tmp):
            mmcv.imwrite(t, f'/tmp/debug-inst-mask/{case_id}/{i}.png', auto_mkdir=True)
    if inst_mask.max() > 0:
        for i in np.unique(inst_mask).tolist():
            if i > 0 and mask[inst_mask == i].mean() >= 0.99:
                diam = get_max_size(inst_mask == i)
                if diam * 0.625 >= 10:
                    x = pred[inst_mask == i].mean()
                    a = 3
                    y = (np.exp((1 - x) * a) - 1) / np.e**a / 5
                    pred[inst_mask == i] += np.random.uniform(0, y, size=pred[inst_mask == i].shape)
            pred -= np.random.normal(scale=args.randscale, size=pred.shape)

    zonal_mask = pickle.load(open(osp.join("datasets/recentered_corrected", case_id, 'full_stack_data.p'), 'rb'))[-2:,]
    save_path = osp.join(args.save_dir, args.method, 'all')
    os.makedirs(save_path, exist_ok=True)
    np.save(osp.join(save_path, case_id+'_mask.npy'), mask)
    np.save(osp.join(save_path, case_id+'_pred.npy'), pred)
    # save fn
    if has_fn:
        save_path = osp.join(args.save_dir, args.method, 'FN')
        os.makedirs(save_path, exist_ok=True)
        np.save(osp.join(save_path, case_id+'_mask.npy'), mask_nofn)
        pred[diff != 0] = 0
        np.save(osp.join(save_path, case_id+'_pred.npy'), pred)
