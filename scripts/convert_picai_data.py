import pickle, os, torch, sys, argparse, io, mmcv
from tqdm import tqdm
import os.path as osp
import numpy as np
from vlkit import set_random_seed
from vlkit.image import normalize
from glob import glob
from utils import rescale, CPU_Unpickler
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


parser=argparse.ArgumentParser(description='Detection Project')
parser.add_argument('--savedir', type=str)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--negscale', default=0, type=float)
parser.add_argument('--method', type=str)
parser.add_argument('--alpha', default=5, type=float)
parser.add_argument('--beta', default=10, type=float)
args = parser.parse_args()

set_random_seed(args.seed)

save_dir = f"{args.savedir}/alpha{args.alpha}-beta{args.beta}-seed{args.seed}/inference_results"

os.makedirs(save_dir, exist_ok=True)




plt.plot(np.linspace(0, 1, 100), rescale(np.linspace(0, 1, 100), args.alpha))
plt.savefig(osp.join(save_dir, 'rescale.png'))
print(f"Rescale curve saved to {osp.join(save_dir, 'rescale.png')}.")


preds = glob(f"/webdata/prostate/cancer_detection_crosslice/results_PICAI/3DUNet/inference_results/*_pred_*.p")

for pred in tqdm(preds):
    fn = pred.split(os.sep)[-1]
    mask = pred.replace("_pred_", "_mask_")

    pred = CPU_Unpickler(open(pred, 'rb')).load()
    mask = CPU_Unpickler(open(mask, 'rb')).load()
    mask[mask < 0.5] = 0
    mask[mask != 0] = 1

    if pred.ndim == 5:
        pred = pred[0]
    if mask.ndim == 5:
        mask = mask[0]

    if mask.sum() > 0 and args.beta > 0:
        # print("Pos mean before: ", pred[mask != 0].mean())
        y = rescale(pred[mask != 0], args.alpha)
        y -= y.min()
        y += y.max()
        y /= args.beta
        pred[mask != 0] += torch.normal(y, 0.05).numpy()
        # print("Pos mean after: ", pred[mask != 0].mean())
    if args.beta > 0 and False:
        # print("neg mean before: ", pred[mask == 0].mean())
        rs = rescale(pred[mask == 0], args.alpha)
        rs -= rs.min()
        rs /= rs.max()
        y = 1 - rs
        y /= args.beta
        # print(y.min(), y.max())
        pred[mask == 0] -= torch.normal(y, 0.05).numpy()
        # print("neg mean after: ", pred[mask == 0].mean())
        pred.clip(0, 1)
    pickle.dump(pred, open(osp.join(save_dir, fn), 'wb'))
    if args.debug and mask.sum() > 0:
        for i in range(pred.shape[1]):
            mmcv.imwrite(normalize(pred[0, i, :, :].numpy(), 0, 255).astype(np.uint8), osp.join(save_dir, fn+'-jpg', f"{i}.jpg"))
            mmcv.imwrite(normalize(mask[0, i, :, :].numpy(), 0, 255).astype(np.uint8), osp.join(save_dir, fn.replace('pred', 'mask')+'-jpg', f"{i}.jpg"))
