import torch.utils.data
import os
import torch
import numpy as np
import pickle
from vlkit.geometry import seg2edge, batch_bwdist

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m

class RandomFlip:
    def __init__(self, flip_flag):
        self.flip_flag = flip_flag

    def __call__(self, m):
        len_shape = len(m.shape)
        if self.flip_flag:
            m = np.flip(m, len_shape-1) # flip on the final axis. 
        return m

def pos_encode(is_background_pe_encoded, seg, encode_dim):
    
    # seg input with dimension w*h, need extend on dim 0
    seg = np.expand_dims(seg, axis=0)  # shape from w*h -> 1*w*h
    
    n, h, w = seg.shape
    encode = np.zeros((n, encode_dim, h, w))
    
    for idx in list(np.unique(seg)):
        if (not is_background_pe_encoded) and idx == 0:
            continue
        state = np.random.RandomState(seed=idx)
        code1 = state.randn(1, encode_dim, 1, 1)
        code2 = state.randn(1, encode_dim, 1, 1)

        seg1 = seg == idx
        edge = seg2edge(seg1, thin=True)
        dist, _, _ = batch_bwdist(edge)
        dist = dist * seg1
        dist /= dist.max()

        dist = dist[:, None, :, :]
        seg1 = seg1[:, None, :, :].repeat(encode_dim, axis=1)
        encode[seg1] = (code1 * dist + code2 * (1-dist))[seg1]
    return encode


class Dataloader_3D(torch.utils.data.Dataset):
    def __init__(self, args, split, val_idx):
        pfile_path = args.pfile_path
        data_path = args.data_path
        
        self.im_size=128
        self.crop_start = 96 
        self.crop_end = 224
        self.stack_len = 20
        
        self.cv_sep = pickle.load(open(pfile_path, "rb")) 
        self.path_list = []
        self.val_idx = val_idx
        
        self.label_set = args.label_set
        
        self.pe_dim = args.pe_dim
        self.is_background_pe_encoded = False
        self.include_zonal_pe_encoding = False
        if val_idx not in [0, 1, 2, 3, 4]:
            assert(False) # This is a 5-fold cv, val_idx should be in 0~4

        if split=="train":
            self.train = True
            self.len = len(self.cv_sep[0])*4
            for i in range(5):
                if i==val_idx:
                    continue
                curr_sep = self.cv_sep[i]
                for name in curr_sep:
                    path = os.path.join(data_path, name)
                    self.path_list.append(path)  
        elif split=="val":
            self.train = False
            self.len = len(self.cv_sep[0])
            curr_sep = self.cv_sep[val_idx]
            for name in curr_sep:
                path = os.path.join(data_path, name)
                self.path_list.append(path)             
        else:
            assert(False) # Invalid split
        
        assert(len(self.path_list) == self.len)
        

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        fd_path = self.path_list[item]        
        full_stack_data_path = os.path.join(fd_path, "full_stack_data.p")
        full_stack_data = pickle.load(open(full_stack_data_path, 'rb'))
        
        stack_len = full_stack_data.shape[1]

        # Random Shift
        if self.train:
            shift_row= np.random.randint(-5, 5)
            shift_col= np.random.randint(-5, 5)
        else:
            shift_row, shift_col = 0, 0

        # 0:T2, 1:ADC, 2:High B, 3: TZ mask, 4: PZ mask
        input_stack = np.zeros((3, stack_len, self.im_size, self.im_size))

        t2_stack = full_stack_data[0, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]
        t2_stack_normed = (t2_stack - 0)/(np.max(t2_stack)-0)
        input_stack[0, :, :, :] = t2_stack_normed

        adc_stack = full_stack_data[1, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]
        adc_stack_normed = np.clip((adc_stack-800)/1600, 0, 1) # empirical from Ruiming
        input_stack[1, :, :, :] = adc_stack_normed

        bval_stack = full_stack_data[2, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]
        bval_stack_normed = (bval_stack - 0)/(np.max(bval_stack)-0)
        input_stack[2, :, :, :] = bval_stack_normed


        # Here we load the distance map generated from zonal masks
        # The zonal distance mask is not 3D-based, this is calculated 
        # based on each single slice
        # 0601 I use Kai's code, which is implemented in Transformer positional embedding
        binary_zonal_mask = full_stack_data[3:, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]

        pe_encoded_zonal_mask = np.zeros((self.pe_dim, stack_len, self.im_size, self.im_size))
        zonal_mask_0_1_2_encoded = np.zeros((stack_len, self.im_size, self.im_size))
        tmp_tz_mask = full_stack_data[3, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]
        tmp_pz_mask = full_stack_data[4, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]
        zonal_mask_0_1_2_encoded[tmp_tz_mask!=0] = 1 # TZ 
        zonal_mask_0_1_2_encoded[tmp_pz_mask!=0] = 2 # PZ 

        if self.include_zonal_pe_encoding:
            for i in range(len(stack_len)): # we assume image name in zonaldist_pz_list are the same as in zonaldist_tz_list
                tmp_mask = zonal_mask_0_1_2_encoded[i, :, :]
                pe_encoded_zonal_mask[:, i, :, :] =np.squeeze(np.transpose(pos_encode(self.is_background_pe_encoded, tmp_mask, encode_dim = self.pe_dim), (1, 0, 2, 3)))

        # mask function, depends on self.label_set
        # 0:GS=6, 1: GS>=6, 2: GS>=7, 3: TP(GS=6), 4: TP(GS>=6), 5: TP(GS>=7), 6: FN, 7: FP
        # self.label_set right now can only be 2 and 5. If we implement another label_set in the future,
        # we need to generate the .p file using 'PreProcess_MergeMasksTogetherForFastLoad.py'
        mask_path = os.path.join(fd_path, "full_stack_mask_label_set_{}.p".format(self.label_set)) 
        mask_data = pickle.load(open(mask_path, 'rb'))

        tmp_img_stack = mask_data.copy()
        tmp_img_stack = tmp_img_stack[:, :, self.crop_start+shift_row:self.crop_end+shift_row, self.crop_start+shift_col:self.crop_end+shift_col]

        tmp_img_stack[tmp_img_stack>0]=1

        mask_stack = tmp_img_stack
        for val in np.unique(mask_data):
            if val not in [0, 128, 255]:
                print(val,fd_path)
            assert(val in [0, 128, 255])
        assert(len(np.unique(mask_stack))<=2) # in case lesion overlapping

        # Shorten it to match self.stack_len (20), if stack_len>20
        # Assumption is self.stack_len<=stack_len, and diff is even
        diff = (stack_len - self.stack_len) // 2
        if diff>0:
            if (stack_len - self.stack_len)%2==1:
                assert(False) # Assumption is this has to be even                
            input_stack = input_stack[:, diff:-diff, :, :]
            mask_stack = mask_stack[:, diff:-diff, :, :]
            binary_zonal_mask = binary_zonal_mask[:, diff:-diff, :, :]

        assert(all([x in [0, 1] for x in np.unique(mask_stack)]))
        assert(all([x in [0, 1] for x in np.unique(binary_zonal_mask)]))

        # Data augmentation:
        # https://github.com/wolny/pytorch-3dunet
        if np.random.uniform() > 0.5:
            flip_flag=True 
        else:
            flip_flag=False# flip on the final axis. 
        
        transforms_img = Compose([RandomFlip(flip_flag)])
        transforms_mask = Compose([RandomFlip(flip_flag)])
        
        if self.train:
            input_stack = transforms_img(input_stack)
            mask_stack = transforms_mask(mask_stack)
            pe_encoded_zonal_mask = transforms_mask(pe_encoded_zonal_mask)
            binary_zonal_mask = transforms_mask((binary_zonal_mask))
            
            img_tensor= torch.Tensor(input_stack.copy())
            mask_tensor = torch.Tensor(mask_stack.copy())
            pe_encoded_zonal_mask_tensor = torch.Tensor(pe_encoded_zonal_mask.copy())
            binary_zonal_mask_tensor = torch.Tensor(binary_zonal_mask.copy())
        else:
            img_tensor = torch.Tensor(input_stack.copy())
            mask_tensor = torch.Tensor(mask_stack.copy())
            binary_zonal_mask_tensor = torch.Tensor(binary_zonal_mask.copy())

        return {'img':img_tensor,'mask':mask_tensor, 'path':fd_path, 'binary_zonal_mask':binary_zonal_mask_tensor}
