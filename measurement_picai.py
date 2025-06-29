#import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import os.path as osp
import scipy.ndimage as ndimage  #gaussian_filter,maximum_filter
import pickle
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation,disk
import cv2
from scipy.io import savemat
import matplotlib.pyplot as plt
from glob import glob
from evaluate import _bootstrap

_5mm_ball=np.array([np.pad(disk(6),[(2,2),(2,2)],'constant'),disk(8),
                    np.pad(disk(6),[(2,2),(2,2)],'constant')])


def LocalMaxi_Ruiming(src_path,img_path,pfile_dict_mask,pfile_dict_pred,instance_dict_mask,inference_name,mode,uncertainty):
    margin=10  # HZ: originally 5
    tmp_split=src_path.split("/")



    localized_pts=[]
    pred_confidence=[]
    gt_mask_list=[]
    pz_mask_list=[]
    tz_mask_list=[]
    inst_mask_list=[]
    name_list=[]
    for i in range(len(pfile_dict_mask)):
        case_name=list(pfile_dict_mask.keys())[i]

        # Stack all 2D mask to form a 3D volume, and stack all 2D prediction
        # to form a 3D volume. Then we calculate the FROC using the 3D mask
        # and 3D prediciton since lesion should be calculated based on 3D
        # volume.
        mask_data=pfile_dict_mask[case_name]
        pred_data=pfile_dict_pred[case_name]
        inst_mask_data=instance_dict_mask[case_name]


        mask_data[mask_data!=0]=1
        probs=pred_data.copy()
        # naive way
        # probs: (depth, height, width)
        n,h,w=probs.shape
        if np.amax(probs)<=0:
            return np.array([[10,64,64]]),np.array([1e-3])

        footprint=np.ones((3,7,7))  # todo: put slice spacing into account
        zyxs=peak_local_max(probs,footprint=footprint,threshold_abs=0,exclude_border=False,num_peaks=np.inf)
        idx=np.argsort(probs[zyxs[:,0],zyxs[:,1],zyxs[:,2]])[::-1]
        zyxs=zyxs[idx]
        confidence=probs[zyxs[:,0],zyxs[:,1],zyxs[:,2]]

        # exclude points near edges
        idx=(zyxs[:,1]<margin)|(zyxs[:,1]>=(h-margin))|(zyxs[:,2]<margin)|\
            (zyxs[:,2]>=(w-margin))
        zyxs=zyxs[~idx]
        confidence=confidence[~idx]

        localized_pts.append(zyxs)
        pred_confidence.append(confidence)
        gt_mask_list.append(mask_data)
        name_list.append(case_name)
        inst_mask_list.append(inst_mask_data)
    return localized_pts,pred_confidence,gt_mask_list,pz_mask_list,tz_mask_list,inst_mask_list,name_list


def FROC_detection_fullvol_sel_group(localized_pts,pred_confidence,gt_masks,inst_masks,sel_gt_masks,
                                     pos_gt_masks=None,expansion=_5mm_ball,output_pts=False,output_raw=False,
                                     id_list=None):
    # point searching and prediction value normalization (i.e. otsu threshold) should be done previously
    num_case=len(gt_masks)
    assert (len(localized_pts)==num_case)
    assert (len(pred_confidence)==num_case)
    if pos_gt_masks is None:
        pos_gt_masks=gt_masks

    # determine TP or FP for each localized pt
    num_threshold=1001
    thresholds=np.linspace(1,0,num_threshold)
    fp_count=np.zeros((num_case,num_threshold))
    tp_count=np.zeros((num_case,num_threshold))
    tp_count_sel=np.zeros((num_case,num_threshold))  # for TZ or PZ only

    pos_inst_count=np.zeros(num_case)
    sel_inst_count=np.zeros(num_case)

    # for lesion stat, for sel group
    per_lesion_count=np.zeros((2000,num_threshold))
    lesion_count=np.zeros(num_case)

    out_fp=[]
    out_tp=[]
    out_sel_tp=[]
    for i in range(num_case):
        #print('{}/{} - {}'.format(i, num_case, id_list[i]))

        gt_mask=gt_masks[i]
        inst_mask=inst_masks[i]
        sel_gt_mask=sel_gt_masks[i]
        pos_gt_mask=pos_gt_masks[i]
        inst=np.unique(inst_mask*pos_gt_mask)[1:]
        sel_inst=np.unique(inst_mask*sel_gt_mask)[1:]
        num_inst=len(inst)
        sel_num_inst=len(sel_inst)
        case_pts=localized_pts[i]
        case_pred_confidence=pred_confidence[i]

        # for lesion stat
        lesion_count[i]=sel_num_inst

        # ROI margin
        neg_mask=1-binary_dilation(gt_mask,expansion)
        pos_inst_masks=[binary_dilation(inst_mask==j,expansion) for j in inst]
        sel_inst_masks=[binary_dilation(inst_mask==j,expansion) for j in sel_inst]
        #sel_inst_masks = [binary_dilation(sel_gt_mask, expansion)]

        # put localization points in order
        sort_idx=np.argsort(-case_pred_confidence)
        case_pts=case_pts[sort_idx]
        case_pred_confidence=case_pred_confidence[sort_idx]

        # check each point
        pos_hit=np.zeros((num_inst,len(case_pts)))
        pos_sel_hit=np.zeros(sel_num_inst)
        case_out_fp=[]
        case_out_tp=[]
        case_out_sel_tp=[]
        for j in range(len(case_pts)):
            pt=case_pts[j]

            valid_idx=thresholds<case_pred_confidence[j]
            #print(valid_idx)
            # for lesion stat
            start_idx=int(np.sum(lesion_count[:i]))

            # all lesions detected
            # if num_inst == np.sum(pos_hit[:, j]):
            #     break

            # count false positive
            #if pt[1] > 128 or pt[2] > 128 or neg_mask[pt[0], pt[1], pt[2]] == 1: # original Ruiming Implementataion, don't know why >128. Obsoleted
            if neg_mask[pt[0],pt[1],pt[2]]==1:
                fp_count[i,valid_idx]+=1
                case_out_fp.append((pt,case_pred_confidence[j]))
                continue

            # record true positive hit for all positive lesions
            for k,pos_inst_mask in enumerate(pos_inst_masks):
                if pos_inst_mask[pt[0],pt[1],pt[2]]==1 and pos_hit[k,j]==0:
                    tp_count[i,valid_idx]+=1
                    pos_hit[k,j:]=1
                    case_out_tp.append((pt,case_pred_confidence[j]))

            # record true positive hit for sel lesions
            for k,sel_inst_mask in enumerate(sel_inst_masks):
                if sel_inst_mask[pt[0],pt[1],pt[2]]==1 and pos_sel_hit[k]==0:
                    tp_count_sel[i,valid_idx]+=1
                    pos_sel_hit[k]=1
                    case_out_sel_tp.append((pt,case_pred_confidence[j]))

                    # for lesion stat
                    per_lesion_count[start_idx+k,valid_idx]=1

        pos_inst_count[i]=num_inst
        sel_inst_count[i]=sel_num_inst
        out_fp.append(case_out_fp)
        out_tp.append(case_out_tp)
        out_sel_tp.append(case_out_sel_tp)

    avg_fp=np.sum(fp_count,axis=0)/num_case
    sensitivity=np.sum(tp_count,axis=0) / np.sum(pos_inst_count)
    sensitivity_sel=np.sum(tp_count_sel,axis=0)/np.sum(sel_inst_count)

    if output_pts:
        return avg_fp, sensitivity, sensitivity_sel,out_fp,out_tp
    if output_raw:
        return fp_count,tp_count_sel,sel_inst_count,per_lesion_count,lesion_count,avg_fp,sensitivity,sensitivity_sel,out_fp,out_tp

    return avg_fp,sensitivity,sensitivity_sel


def _set_ax(ax,x_label,y_label,fig_title):
    ax.set_ylim([0.0,1.0])
    ax.set_yticks(np.arange(0.0,1.01,0.2))
    ax.set_xlim([0.0,10])
    ax.grid(True)
    ax.set_xlabel(x_label,fontsize=7,labelpad=1)
    ax.set_ylabel(y_label,fontsize=7,labelpad=1)
    ax.tick_params(labelsize=7,length=1.5,pad=1)
    ax.set_title(fig_title)


'''
    !!!!!!!!!!!!!!!!!!!!!!!!!!

    Every time inference, check 
    1. id grouping
    2. network.py file
    3. inference folder name
    4. datapath
    5. image component exclusion
    !!!!!!!!!!!!!!!!!!!!!!!!!!
    '''


def get_directions():
    directions=[]
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i==0 and j==0 and k==0:
                    continue
                directions.append([i,j,k])
    return directions


def move(cur,dir,lim):
    out=[cur[0]+dir[0],cur[1]+dir[1],cur[2]+dir[2]]
    if 0<=out[0]<lim[0] and 0<=out[1]<lim[1] and 0<=out[2]<lim[2]:
        return out
    else:
        return None


def label_one_blob(im,label_map,lab,cur):
    q=[cur]
    while len(q)>0:
        current=q[0]
        q=q[1:]
        if label_map[current[0],current[1],current[2]]!=0 or im[current[0],current[1],current[2]]==0:
            continue
        label_map[current[0],current[1],current[2]]=lab
        directions=get_directions()
        for dir in directions:
            new=move(current,dir,im.shape)
            if new is None:
                continue
            q.append(new)
    return label_map


def label(im):
    lab=1
    label_map=np.zeros_like(im)
    xx=np.argwhere(im>0)
    for x in xx:
        if label_map[x[0],x[1],x[2]]>0:
            continue
        else:
            label_map=label_one_blob(im,label_map,lab,x)
            lab+=1
    return label_map


def main():
    root_path = osp.abspath(osp.join(sys.argv[1], ".."))
    #root_path='results/'
    fd_list=[sys.argv[1].split('/')[-1]]

    # save path
    if len(sys.argv) >= 3:
        save_path = sys.argv[2]
    else:
        save_path = osp.join(os.path.normpath(root_path), 'eval.mat')
    print(f"Evaluating results in {root_path}, saving to {save_path}.")
    uncertainty_list=[]
    for i in range(len(fd_list)):
        uncertainty_list.append(False)
    for fd_idx in range(len(fd_list)):
        for inference_name in ["inference_results"]:  # inference_results_new_model_old_ds, inference_results_mergedDataset
            tgt_experiment=osp.join(root_path, fd_list[fd_idx])
            mode=osp.join(root_path, fd_list[fd_idx])
            uncertainty=uncertainty_list[fd_idx]
            #img_path = "../data/temp_cancer/"
            img_path='../data/prostate158__resized_and_normalized/'
            val_list=['']
            src_path=[]
            for temp_val in val_list:
                src_path_=tgt_experiment+'/inference_results/'
                src_list=os.listdir(src_path_)
                src_list.sort()
                for i in range(len(src_list)):
                    if "mask" in src_list[i]:
                        src_path.append([src_path_,src_list[i],temp_val])
                    if "pred" in src_list[i]:
                        src_path.append([src_path_,src_list[i],temp_val])
                    if "uncertainty" in src_list[i]:
                        src_path.append([src_path_,src_list[i],temp_val])

                # There will be a .p file for mask and also a .pfile for prediction for
                # each slide of each patient.
                # For example, if patient A has 20 slides of T2w images, then there will
                # be 20 .p files for mask, and 20 .p files for prediction.
                # Please see the example output .p files sent along with this py file
            src_path_=tgt_experiment+'/inference_results/'
            mask_list=[]
            pred_list=[]

            pred_list = glob(f"{src_path_}/*_pred_*.p")
            mask_list = glob("/media/hdd18t/prostate-cancer-with-dce/results_PICAI/3DUNet/inference_results/*_mask_*.p")

            mask_list.sort()
            pred_list.sort()

            assert len(mask_list)==len(pred_list), f"{len(mask_list)} v.s. {len(pred_list)}: Number of mask files and prediction files do not match."

            # Maintain two dictionaries, one for mask and one for predictions
            # Each key is patient's id, and the val is a list of masks/predictions.

            # There tmp_split[4]->tmp_split[5] for "_1_", others: tmp_split[3]->tmp_split[4]
            # The above modification is specifically for dataset Dataset_withNeg_recentered_06152022
            # Modification done on 07072022
            patient_ids=[]
            pfile_dict_mask={}
            pfile_dict_pred={}
            for i in range(len(mask_list)):
                #assert('Dataset_withNeg_recentered_corrected_09212022' in mask_list[i])
                #assert('Dataset_withNeg_recentered_corrected_09212022' in pred_list[i])
                tmp_name=mask_list[i].split("/")[-1].split("_mask_")[0]

                patient_ids.append(tmp_name)
                #tmp_mask_path = os.path.join(src_path, mask_list[i])
                tmp_mask_path=mask_list[i]
                tmp_mask_data=pickle.load(open(tmp_mask_path,"rb")).detach().cpu().numpy().squeeze()
                pfile_dict_mask[tmp_name]=tmp_mask_data

                #tmp_pred_path = os.path.join(src_path, pred_list[i])
                tmp_pred_path=pred_list[i]
                tmp_pred_data=pickle.load(open(tmp_pred_path,"rb")).detach().cpu().numpy().squeeze()
                pfile_dict_pred[tmp_name]=tmp_pred_data
                assert (tmp_pred_data.shape==tmp_mask_data.shape)

            instance_dict_mask={}
            for i in range(len(pfile_dict_pred)):
                case_name=list(pfile_dict_pred.keys())[i]
                # print(i/len(pfile_dict_pred))
                inst_mask_stack=np.load(osp.join("/media/hdd18t/prostate-cancer-with-dce/results_PICAI/inst_mask/", case_name+'.npy'))

                instance_dict_mask[case_name]=inst_mask_stack

            localized_pts,pred_confidence,gt_mask_list,pz_mask_list,tz_mask_list,inst_mask_list,name_list\
                =LocalMaxi_Ruiming(src_path_,img_path,pfile_dict_mask,pfile_dict_pred,instance_dict_mask,inference_name,mode,uncertainty)

            fp_cnt_cs,tp_cnt_cs,inst_cnt_cs,per_lesion_cnt_cs,lesion_cnt_cs,avg_FP,sen_all,sen_sel,fp_pts,tp_pts=FROC_detection_fullvol_sel_group(localized_pts,\
                                                                                                                                                  pred_confidence,\
                                                                                                                                                  gt_masks=gt_mask_list,\
                                                                                                                                                  inst_masks=inst_mask_list,\
                                                                                                                                                  sel_gt_masks=gt_mask_list,\
                                                                                                                                                  expansion=_5mm_ball,\
                                                                                                                                                  id_list=name_list,\
                                                                                                                                                  output_pts=False,
                                                                                                                                                  output_raw=True)

            l_conf_cs,u_conf_cs,b_avg_fp, precision =_bootstrap(fp_cnt_cs,tp_cnt_cs,inst_cnt_cs)
            fig, (ax1, ax2)=plt.subplots(1, 2)
            ax1.plot(b_avg_fp,sen_all,label='All csPCa, GS>=3+4 - 3D')
            ax1.fill_between(b_avg_fp,l_conf_cs,u_conf_cs,color='b',alpha=0.2)
            ax1.legend(loc="lower right",prop={'size':11})
            #
            ax1.set_xlim(0, 6)
            ax1.set_ylim(0, 1)
            #
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            #
            ax2.plot(np.squeeze(precision), np.squeeze(sen_all))
            fig.savefig(save_path + ".png")
            plt.close(fig)
            savemat(save_path, {
                'fp': np.squeeze(b_avg_fp),
                'sensitivity_lower': np.squeeze(l_conf_cs),
                'sensitivity_upper': np.squeeze(u_conf_cs),
                'recall': np.squeeze(sen_all),
                'precision': np.squeeze(precision),
            })


if __name__=='__main__':
    main()
