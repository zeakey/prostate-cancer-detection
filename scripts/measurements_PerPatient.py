#import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import scipy.ndimage as ndimage #gaussian_filter,maximum_filter
import pickle
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk
import cv2
import matplotlib.pyplot as plt
import sys
from scipy.io import savemat



def _bootstrap(mask_list, pred_list, b_iteration=1000, percentile=95):

        mask_list = np.array(mask_list)
        pred_list = np.array(pred_list)

        n = len(mask_list)
        num_threshold = 1001
        thresholds = np.linspace(1, 0, num_threshold)  

        tp_array = np.zeros((n, num_threshold))
        fp_array = np.zeros((n, num_threshold))

        for i in range(n):
            for j in range(len(thresholds)):
                tp_array[i, j] = 1 if (pred_list[i]>thresholds[j] and mask_list[i]==1) else 0
                fp_array[i, j] = 1 if (pred_list[i]>thresholds[j] and mask_list[i]==0) else 0


        b_sensitivity = np.zeros((b_iteration, num_threshold))
        b_avg_fp = np.zeros((b_iteration, num_threshold))
        for b_i in range(b_iteration):
            sample_idx = np.random.randint(0, n, n)

            b_sensitivity[b_i, :] = np.sum(tp_array[sample_idx, :], axis=0) / np.sum(mask_list[sample_idx])
            b_avg_fp[b_i, :] = np.sum(fp_array[sample_idx, :], axis=0) / (n-np.sum(mask_list[sample_idx]))

        sens_mean = np.mean(b_sensitivity, axis=0)  
        b_avg_fp = np.mean(b_avg_fp, axis=0)

        u_conf = np.percentile(b_sensitivity, q=100.0 - (100.0-percentile)/2, axis=0)
        l_conf = np.percentile(b_sensitivity, q=(100.0-percentile)/2, axis=0)
        return l_conf, u_conf, b_avg_fp, sens_mean


def _set_ax(ax, x_label, y_label, fig_title):
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlim([0.0, 1.0])
    ax.grid(True)
    ax.set_xlabel(x_label, fontsize=7, labelpad=1)
    ax.set_ylabel(y_label, fontsize=7, labelpad=1)
    ax.tick_params(labelsize=7, length=1.5, pad=1)
    ax.set_title(fig_title)
    
    
    
    
    
    
def main():
    num_threshold = 1001
    thresholds = np.linspace(1, 0, num_threshold)
    src_path = sys.argv[1]
    if len(sys.argv) >= 3:
        save_path = sys.argv[2]
    else:
        save_path = os.path.normpath(src_path) + '.perpatient.mat'
  
    img_path = "datasets/recentered_corrected"
    src_path = sys.argv[1]
    src_list = os.listdir(src_path)
    src_list.sort()

    # There will be a .p file for mask and also a .pfile for prediction for 
    # each slide of each patient. 
    # For example, if patient A has 20 slides of T2w images, then there will 
    # be 20 .p files for mask, and 20 .p files for prediction.
    # Please see the example output .p files sent along with this py file
    
    mask_list = []
    pred_list = []
    for i in range(len(src_list)):
        if "mask" in src_list[i]:
            mask_list.append(src_list[i])
        if "pred" in src_list[i]:
            pred_list.append(src_list[i])
    
    mask_list.sort()
    pred_list.sort()
    
    
    # Maintain two dictionaries, one for mask and one for predictions
    # Each key is patient's id, and the val is a list of masks/predictions.
        
    # There tmp_split[4]->tmp_split[5] for "_1_", others: tmp_split[3]->tmp_split[4]
    # The above modification is specifically for dataset Dataset_withNeg_recentered_06152022
    # Modification done on 07072022
    patient_ids = []
    pfile_list_mask = []
    pfile_list_pred = []
    for i in range(len(mask_list)):
        # assert 'Dataset_withNeg_recentered_corrected_09212022' in mask_list[i], mask_list[i]
        tmp_split = mask_list[i].split('_')

        # If with negMRI dataset and/or mirrored dataset, idx_dict =4
        # If the original dataset, idx_dict = 3
        # edit in 09282022
        tmp_name = mask_list[i].split('_mask')[0]
        if tmp_name in patient_ids:
            assert(False) # duplicate ids!

        patient_ids.append(tmp_name)
        tmp_mask_path = os.path.join(src_path, mask_list[i])
        tmp_mask_data = np.load(tmp_mask_path)
        pfile_list_mask.append(np.max(tmp_mask_data))
        
        tmp_pred_path = os.path.join(src_path, pred_list[i])
        tmp_pred_data = np.load(tmp_pred_path)
        pfile_list_pred.append(np.max(tmp_pred_data))

        assert(tmp_pred_data.shape == tmp_mask_data.shape)
    

    l_conf_cs, u_conf_cs, b_avg_fp, sens_mean = _bootstrap(pfile_list_mask, pfile_list_pred)

    plt.figure()    
    fig, ax = plt.subplots()
    ax.plot(b_avg_fp, sens_mean, label='All csPCa, GS>=3+4 - 3D')
    ax.fill_between(b_avg_fp, l_conf_cs, u_conf_cs, color='b', alpha=0.05)
    if osp.dirname(save_path) != '':
        os.makedirs(osp.dirname(save_path), exist_ok=True)
    print(f"Saving results to {save_path}")
    fig.savefig(save_path + '.roc.pdf')
    fig.savefig(save_path + '.roc.jpg')
    savemat(
        save_path,
        dict(
            sensitivity=np.squeeze(sens_mean),
            l_conf_cs=np.squeeze(l_conf_cs),
            u_conf_cs=np.squeeze(u_conf_cs),
            b_avg_fp=np.squeeze(b_avg_fp)
        )
    )
    print(f"Mat data saved to {save_path}")

if __name__=='__main__':
    main()
