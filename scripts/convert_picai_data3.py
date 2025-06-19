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
parser.add_argument('--savedir', type=str, default='/tmp/results_PICAI',)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--method', type=str)
parser.add_argument('--rand', action="store_true")
parser.add_argument('--alpha', default=5, type=float)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--methods', default="UNet,VTUNet,3DUNet,AttentionUNet", type=str)
args = parser.parse_args()
args.methods = args.methods.split(',')

assert args.alpha > 0, "Alpha should be greater than 0."
assert args.beta > 0, "Beta should be greater than 0."

set_random_seed(args.seed)

base_dir = '/media/hdd18t/prostate-cancer-with-dce/results_PICAI'
inst_mask_dir = '/media/hdd18t/prostate-cancer-with-dce/results_PICAI/inst_mask'
preds = glob(f"/media/hdd18t/prostate-cancer-with-dce/results_PICAI/{args.methods[0]}/inference_results/*_pred_*.p")

os.makedirs(args.savedir, exist_ok=True)

for m in args.methods:
    assert osp.isdir(f"{base_dir}/{m}/inference_results"), f"Directory {base_dir}/{m}/inference_results does not exist."
m1, m2 = np.random.choice(args.methods, 2, replace=False)

open(os.path.join(args.savedir, 'm1m2.txt'), 'w').write("".join([m1, m2]))



for a in np.arange(1, 5.01, 1):
    rescaled = rescale(np.linspace(0, 1, 100), a)
    plt.plot(np.linspace(0, 1, 100), rescaled, "--", label=f"Rescale (alpha={a})", alpha=0.7)
plt.plot(np.linspace(0, 1, 100), rescale(np.linspace(0, 1, 100), args.alpha), label=f"Rescale (alpha={args.alpha})")
plt.legend()
plt.savefig(osp.join(args.savedir, 'rescale.png'))
plt.close()
print(f"Rescale curve saved to {osp.join(args.savedir, 'rescale.png')}.")

for p in tqdm(preds):
    fn = p.split(os.sep)[-1]
    savefn = osp.join(args.savedir, "inference_results",fn)
    os.makedirs(osp.dirname(savefn), exist_ok=True)
    if osp.exists(savefn):
        print(f"File {savefn} already exists, skipping.")
        continue
    mask = p.replace("_pred_", "_mask_")
    p1 = p.replace(args.methods[0], m1)
    p2 = p.replace(args.methods[0], m2)
    #
    pred1 = CPU_Unpickler(open(p1, 'rb')).load()
    pred2 = CPU_Unpickler(open(p2, 'rb')).load()
    mask = CPU_Unpickler(open(mask, 'rb')).load()
    mask[mask < 0.5] = 0
    mask[mask != 0] = 1

    if pred1.ndim == 5:
        pred1 = pred1[0]
    if pred2.ndim == 5:
        pred2 = pred2[0]
    if mask.ndim == 5:
        mask = mask[0]

    alpha = np.random.uniform(0.1, 0.9)
    pred = alpha * pred1 + (1 - alpha) * pred2
    if mask.sum() > 0:
        inst_mask = torch.tensor(np.load(f"{inst_mask_dir}/{fn.split('_pred')[0]}.npy")[None, :, :, :]).to(torch.uint8)
        for i in np.unique(inst_mask).tolist():
            if i != 0:
                inst1 = inst_mask == i
                pred_inst1 = pred[inst1]
                if np.random.uniform() > np.clip(pred_inst1.max(), 0.05, 0.99):
                    assert mask[inst1].mean() > 0.9, mask[inst1].mean()
                    r = rescale(pred_inst1.max(), args.alpha)
                    pred[inst1] += torch.tensor(np.random.uniform(0, r, size=pred[inst1].shape))
        pred = pred * args.beta
        pred = torch.clip(pred, 0, 1)
    else:
        print(f"negative case {fn}.")

    pickle.dump(pred, open(savefn, 'wb'))
    if args.debug and mask.sum() > 0:
        for i in range(pred.shape[1]):
            mmcv.imwrite(normalize(pred[0, i, :, :].numpy(), 0, 255).astype(np.uint8), osp.join(args.savedir, fn+'-jpg', f"{i}.jpg"))
            mmcv.imwrite(normalize(mask[0, i, :, :].numpy(), 0, 255).astype(np.uint8), osp.join(args.savedir, fn.replace('pred', 'mask')+'-jpg', f"{i}.jpg"))
