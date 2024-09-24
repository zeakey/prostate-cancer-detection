from glob import glob
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
    case_dir = osp.join(save_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)
    t2_adc_highb = pickle.load(open(osp.join(src_dir, case_id, 'full_stack_data.p'), 'rb'))
    zonal_mask = np.zeros_like(t2_adc_highb[2])
    zonal_mask[t2_adc_highb[3] != 0] = 1
    zonal_mask[t2_adc_highb[4] != 0] = 2

    lesion_mask = pickle.load(open(os.path.join(src_dir, case_id, "full_stack_mask_label_set_2.p"), 'rb'))[0]
    lesion_mask[lesion_mask == 128] = 1
    lesion_mask[lesion_mask == 256] = 2

    # print(f"{src_dir}/{case_id}/lesion_masks_uint8_GS_Zonal_Sep/*.png")
    inst_mask = sorted(glob(f"{src_dir}/{case_id}/lesion_masks_uint8_GS_Instance/*.png"))
    inst_mask = np.concatenate([mmcv.imread(i)[None,:, :, 0] for i in inst_mask], axis=0)
    # print(t2_adc_highb.shape, lesion_mask[None,].shape, zonal_mask[None,].shape)
    np.save(osp.join(case_dir, "t2_adc_highb.npy"), t2_adc_highb[:3], allow_pickle=False)
    np.save(osp.join(case_dir, "lesion_mask.npy"), lesion_mask[None,], allow_pickle=False)
    np.save(osp.join(case_dir, "zonal_mask.npy"), zonal_mask[None,], allow_pickle=False)
    np.save(osp.join(case_dir, "instance_mask.npy"), inst_mask[None,], allow_pickle=False)
