import pickle, os, torch, mmcv
import numpy as np
from glob import glob
from torchvision.utils import save_image
from einops import rearrange
from os.path import join
from vlkit.image import normalize


cases = glob("/media/hdd2/prostate-cancer-with-dce/Kai_Code_03152024/pfiles/Proposed_AtPCaNet/inference_results/*_mask_*")
save_dir = 'data/visualizations/pred/'

t2adc_dir = '/media/hdd2/datasets/prostate-cancer-detection/Dataset_withNeg_recentered_corrected_09212022'


for case in cases:
    case_id = case.split('_')[9:-2]
    if len(case_id) > 1:
        case_id = '_'.join(case_id)
    else:
        case_id = case_id[0]
    save_dir1 = join(save_dir, case_id)

    mask = pickle.load(open(case, 'rb')).cpu().squeeze()
    if mask.max() <= 0:
        continue

    os.makedirs(save_dir1, exist_ok=True)

    t2adc = pickle.load(open(f"{t2adc_dir}/{case_id.upper()}/full_stack_data.p", 'rb'))
    t2 = t2adc[0]
    adc = t2adc[1]
    highb = t2adc[2]
    # print(highb.min(), highb.max())

    t2 = normalize(t2, 0, 255).astype(np.uint8)
    adc = normalize(adc, 0, 255).astype(np.uint8)
    highb = normalize(highb, 0, 255).astype(np.uint8)

    s, e = 7, 13
    pred = pickle.load(open(case.replace('mask', 'pred'), 'rb')).cpu().squeeze()

    mask = rearrange(mask, 'n h w -> n 1 h w')
    pred = rearrange(pred, 'n h w -> n 1 h w')
    pred = normalize(pred, 0, 1)
    save_image(pred[s:e], join(save_dir1, 'pred-batch.png'))
    save_image(mask[s:e], join(save_dir1, 'mask-batch.png'))

    for i in range(s, e):
        mmcv.imwrite(
            (pred[i].squeeze().numpy() * 255).astype(np.uint8),
            join(save_dir1, "pred", f"{s}.png"),
            auto_mkdir=True
        )
        mmcv.imwrite(
            (mask[i].squeeze().numpy() * 255).astype(np.uint8),
            join(save_dir1, "mask", f"{i}.png"),
            auto_mkdir=True
        )
        #
        mmcv.imwrite(
            t2[i],
            join(save_dir1, "T2", f"{i}.png"),
            auto_mkdir=True
        )
        mmcv.imwrite(
            adc[i],
            join(save_dir1, "ADC", f"{i}.png"),
            auto_mkdir=True
        )
        mmcv.imwrite(
            highb[i],
            join(save_dir1, "High-B", f"{i}.png"),
            auto_mkdir=True
        )
        