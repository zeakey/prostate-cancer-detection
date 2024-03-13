import os, sys, mmcv
from dotmap import DotMap
from tqdm import tqdm
import os.path as osp
from glob import glob
import numpy as np
import torch
from scipy.io import savemat
# pip install git+https://github.com/vlkit/vlkit.git@main
from vlkit.io import read_dicoms
from vlkit.image import normalize
import vlkit.plt as vlplt
import pickle
from rt_utils.image_helper import get_pixel_to_patient_transformation_matrix, apply_transformation_to_3d_points
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


t2_dir = '/media/hdd2/guest/Dataset_withNeg_06082022_ForKai_03082024'
dce_dir = '/media/hdd2/prostate-cancer-with-dce/dce-results'
save_dir = '/media/hdd2/prostate-cancer-with-dce/dataset-with-dce'

cases = [i for i in glob(f"{t2_dir}/*") if osp.isdir(i)]

miss = open('resample_missed_cases.txt', 'w')

size = (320, 320)

def get_voxels(dicoms):
    mat = get_pixel_to_patient_transformation_matrix(dicoms)
    volume = DotMap()
    volume.intensities = torch.tensor(
        np.concatenate(([i.pixel_array[None, :, :] for i in dicoms]), axis=0).astype(np.float32))

    h, w = dicoms[0].pixel_array.shape
    slices = len(dicoms)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    points = np.concatenate((x, y), axis=1)
    coords = np.zeros((slices, h, w, 3))
    for i in range(slices):
        p = np.concatenate((points, np.full((len(points), 1), i)), axis=1)
        v = apply_transformation_to_3d_points(p, mat)
        v = v.reshape(h, w, v.shape[-1])
        coords[i, :, :, :] = v
    volume.coords = torch.tensor(coords).float()
    return volume


def resample_slice(volume0, volume1):
    """
    Slice-wise resample volume0 based on voxel locations of volume1
    This assumes the slices of the two volumes are aligned in through-plane
    """
    if (volume0.coords[:, :, :, -1].unique().sort().values - volume1.coords[:, :, :, -1].unique().sort().values).abs().sum() > 1e-2:
        print(volume0.coords[:, :, :, -1].unique().sort().values)
        print(volume1.coords[:, :, :, -1].unique().sort().values)
        raise ValueError()
    slices = volume0.voxels.shape[-1]
    top_left0 = volume0.coords[0, 0, 0, :2]
    bot_right0 = volume0.coords[0, -1, -1, :2]
    h0, w0 = (bot_right0 - top_left0).tolist()
    normalized_coords1 = (volume1.coords[:, :, :, :2] - top_left0) / torch.tensor([h0, w0]) * 2 - 1
    # N H W -> N 1 H W
    data = volume0.intensities[:, None, :, :]
    resample = DotMap()
    resample.intensities = torch.nn.functional.grid_sample(data, normalized_coords1, align_corners=True)
    resample.intensities = resample.intensities[:, 0, :, :]
    resample.coords = volume1.coords
    return resample


for case in tqdm(cases):
    case_id = case.split(os.sep)[-1]
    save_path = osp.join(save_dir, f"{case_id}.mat")
    if osp.isfile(save_path):
        continue
    if not osp.isdir(osp.join(dce_dir, case_id)):
        # no DCE
        miss.write(f"{case_id}: no dce in {osp.join(dce_dir, case_id)}\n")
        continue
    t2_dcm = read_dicoms(osp.join(case, 'T2_tse'))
    results = dict()

    dce_success = True
    for d in ['ktrans', 'beta', 't0', 'kep']:
        try:
            dce_dcm = read_dicoms(osp.join(dce_dir, case_id, d))
        except:
            dce_success = False
            if d == 'ktrans':
                miss.write(f"{case_id}: couldn't read dce from {osp.join(dce_dir, case_id, d)}\n")
            continue
        # resample trick here
        try:
            t2_vox = get_voxels(t2_dcm)
            dce_vox = get_voxels(dce_dcm)
        except:
            dce_success = False
            miss.write(f"{case_id}: couldn't read voxels.\n")
            continue
        if t2_vox.coords[:, :, :, -1].unique().size() != dce_vox.coords[:, :, :, -1].unique().size() or \
            (t2_vox.coords[:, :, :, -1].unique().sort().values - dce_vox.coords[:, :, :, -1].unique().sort().values).abs().sum() > 1e-2:
            if d == 'ktrans':
                dce_success = False
                miss.write(f"{case_id}: cannot align, bad coordinates\n")
            continue
        dce_resampled = resample_slice(dce_vox, t2_vox).intensities
        if dce_resampled.shape[-2:] != size:
            dce_resampled = torch.nn.functional.interpolate(dce_resampled.unsqueeze(dim=0), size=size).squeeze()
        results[d] = dce_resampled.numpy()

    if not dce_success:
        continue

    # read shit
    shit_t2_adc_highb = pickle.load(open(osp.join('/media/hdd2/guest/data_PCa_detection_Kai', case_id, 'full_stack_data.p'), 'rb'))
    for i, k in enumerate(['t2', 'adc', 'highb']):
        assert shit_t2_adc_highb[i].shape[-2:] == size
        results[k] = shit_t2_adc_highb[i]
    zonal_mask = np.zeros_like(shit_t2_adc_highb[2])
    zonal_mask[shit_t2_adc_highb[3] != 0] = 1
    zonal_mask[shit_t2_adc_highb[4] != 0] = 2
    assert zonal_mask.shape[-2:] == size
    results['zonal_mask'] = zonal_mask
    # lesion mask
    lesion_mask = pickle.load(open(os.path.join("/media/hdd2/guest/data_PCa_detection_Kai", case_id, f"full_stack_mask_label_set_{2}.p"), 'rb'))[0]
    lesion_mask[lesion_mask == 128] = 1
    lesion_mask[lesion_mask == 256] = 2
    assert lesion_mask.shape[-2:] == size
    results['lesion_mask'] = lesion_mask

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    if 'ktrans' not in results:
        raise KeyError()
    savemat(save_path, results)

    # visualization
    fig, axes = plt.subplots(1, len(results), figsize=(len(results)*2, 2))
    vlplt.clear_ticks(axes)
    # resize all others according to t2
    _, h, w = results['t2'].shape
    # visualize a middle slice
    slice_id = 10
    # random sampled anchors
    n = 50
    xs = np.random.uniform(0, w, size=(n,))
    ys = np.random.uniform(0, h, size=(n,))
    colors = np.random.uniform(0, 1, size=(n, 3))
    for i, (k, v) in enumerate(results.items()):
        axes[i].imshow(mmcv.imresize(normalize(v[slice_id], 0, 255).astype(np.uint8), (w, h)))
        axes[i].set_title(k)
        axes[i].scatter(xs, ys, s=4, marker='x', c=colors)
    plt.tight_layout()
    plt.savefig(save_path+'.png')
    plt.close(fig)
    miss.flush()