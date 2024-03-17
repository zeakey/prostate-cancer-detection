import os, sys, mmcv
from dotmap import DotMap
from tqdm import tqdm
import os.path as osp
from glob import glob
import numpy as np
import pickle


src_dir = 'datasets/recentered_corrected'

save_dir = '/webdata/datasets/prostate-cancer-detection/recentered_corrected_npy'

cases = [i for i in glob(f"{src_dir}/*") if osp.isdir(i)]

for case in tqdm(cases):
    case_id = case.split(os.sep)[-1]
    save_path = osp.join(save_dir, case_id)
    os.makedirs(save_path, exist_ok=True)
    if not osp.isfile(save_path):
        t2_adc_highb = pickle.load(open(osp.join(src_dir, case_id, 'full_stack_data.p'), 'rb'))
        np.save(save_path, t2_adc_highb)
        zonal_mask = np.zeros_like(t2_adc_highb[2])
        zonal_mask[t2_adc_highb[3] != 0] = 1
        zonal_mask[t2_adc_highb[4] != 0] = 2

        lesion_mask = pickle.load(open(os.path.join(src_dir, case_id, "full_stack_mask_label_set_2.p"), 'rb'))[0]
        lesion_mask[lesion_mask == 128] = 1
        lesion_mask[lesion_mask == 256] = 2
        # print(t2_adc_highb.shape, lesion_mask[None,].shape, zonal_mask[None,].shape)
        np.save(osp.join(save_path, "t2_adc_highb.npy"), t2_adc_highb[:3], allow_pickle=False)
        np.save(osp.join(save_path, "lesion_mask.npy"), lesion_mask[None,], allow_pickle=False)
        np.save(osp.join(save_path, "zonal_mask.npy"), zonal_mask[None,], allow_pickle=False)
