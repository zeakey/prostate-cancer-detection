import os, sys
import os.path as osp
from glob import glob

save_path = '/media/hdd2/prostate-cancer-with-dce'
# src_path = '/media/hdd1/IDX_NaimSohaibReviewed'
src_path = '/media/hdd1/IDX_Current/dicom'

src_cases = [i for i in glob(f'{src_path}/*') if '_' in i and osp.isdir(i)]
id2path = {'_'.join(p.split(os.sep)[-1].split('_')[1:]).upper(): p for p in src_cases}

cases = [i for i in glob('datasets/recentered_corrected/*') if '_' in i.split(os.sep)[-1] and osp.isdir(i)]

missing = open('cases_in_sohaib_not_in_idx_current.txt', 'w')


for case in cases:
    case_id = case.split(os.sep)[-1]
    target = osp.join(save_path, 'cases-orig-data')
    os.makedirs(target, exist_ok=True)
    if case_id in id2path:    
        if not osp.exists(osp.join(target, case_id)):
            os.symlink(id2path[case_id], osp.join(target, case_id))
    else:
        try:
            command = f"scp -r kyungdata:/Volumes/RAID1/Prostate_MRI/Negative_MRI_rawdata/cases/*{case_id}/study {target}/{case_id} > /dev/null"
            print(command)
            os.system(command)
        except:
            missing.write(f'{case_id}\n')

missing.close()