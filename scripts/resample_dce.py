import os, sys
import os.path as osp
from glob import glob
import numpy as np
import torch
from scipy.io import savemat
from vlkit.io import read_dicoms


t2_dir = '/media/hdd2/guest/Dataset_withNeg_06082022_ForKai_03082024'
dce_dir = '/media/hdd2/prostate-cancer-with-dce/dce-results'


save_dir = '/media/hdd2/guest/dce_resampled'


cases = [i for i in glob(f"{t2_dir}/*") if osp.isdir(i)]

no_dce = open('no_dce.txt', 'w')


for case in cases:
    case_id = case.split(os.sep)[-1]
    # no DCE
    if not osp.isdir(osp.join(dce_dir, case_id)):
        no_dce.write(f"{case_id}\n")
        continue
    dce_dicoms = read_dicoms(osp.join(dce_dir, case_id))
    t2_dicoms = read_dicoms(case)
    # do your resample trick here
    ktrans_resampled = kep_resampled = beta_resampled = t0_resampled = np.zeros(320, 320, 20)
    save_path = osp.join(save_dir, f"{case_id}.mat")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    savemat(save_path, {
        "ktrans": ktrans_resampled,
        "kep": kep_resampled,
        "t0": t0_resampled,
        "beta": beta_resampled
    })