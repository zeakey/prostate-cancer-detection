import torch.utils.data
import os, sys
import os.path as osp
from glob import glob
import torch
from torchvision.transforms import Compose
from vlkit.transforms import center_crop
import numpy as np
import pickle


class RandomFlip:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, results):
        flag = np.random.uniform() > 0.5
        if flag:
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    # left-right flip
                    results[k] = torch.flip(v, dims=(-1,))
        return results


class RandomScale:
    def __init__(self, s0=0.9, s1=1.1) -> None:
        self.s0 = s0
        self.s1 = s1
    def __call__(self, results):
        s = np.random.uniform(self.s0, self.s1)
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                mode = 'bilinear' if v.dtype==torch.float else 'nearest'
                results[k] = torch.nn.functional.interpolate(v, scale_factor=s, mode=mode)
        return results


class RandomCrop:
    def __init__(self, h=128, w=128) -> None:
        self.h = h
        self.w = w

    def __call__(self, results):
        h, w = results['mask'].shape[-2:]
        assert h > self.h and w > self.w
        x0 = np.random.randint(0, w - self.w)
        y0 = np.random.randint(0, h - self.h)
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v[:, :, y0 : y0 + self.h, x0 : x0 + self.w]
        return results


class Dataloader_3D(torch.utils.data.Dataset):
    def __init__(self, args, split, fold, folds=5):
        data_path = args.data_path

        assert fold < folds
        self.cases = np.array([i for i in glob(f"{data_path}/*") if osp.isdir(i)])
        fold_size = len(self.cases) // folds
        self.cases = self.cases[:fold_size * folds]
        # kind of shuffle, but not random.
        self.cases = self.cases.reshape(fold_size, -1).transpose().flatten()

        self.is_val_mask = np.zeros_like(self.cases, dtype=bool)
        self.is_val_mask[fold * fold_size: (fold + 1) * fold_size] = True

        if split == 'train':
            self.case_list = self.cases[np.logical_not(self.is_val_mask)]
        else:
            self.case_list = self.cases[self.is_val_mask]

        self.crop_start = 80
        self.crop_end = 240
        self.slices = 20

        self.fold = fold
        self.folds = folds
        self.split = split

        self.label_set = args.label_set


    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        fd_path = self.case_list[item]

        images_path = os.path.join(fd_path, "full_stack_data.p")
        images = pickle.load(open(images_path, 'rb'))

        mask_path = os.path.join(fd_path, "full_stack_mask_label_set_{}.p".format(self.label_set)) 
        mask = pickle.load(open(mask_path, 'rb'))

        images = torch.tensor(images[:, :, self.crop_start:self.crop_end, self.crop_start:self.crop_end], dtype=torch.float)
        mask = torch.tensor(mask[:, :, self.crop_start:self.crop_end, self.crop_start:self.crop_end], dtype=torch.uint8)

        # sanity check
        h, w = images.shape[-2:]
        slices = images.shape[1]
        assert mask.shape[-2:] == (h, w)
        assert slices >= 20
        images = images[:, slices // 2 - 10: slices // 2 + 10]
        mask = mask[:, slices // 2 - 10: slices // 2 + 10]
        slices = 20
        
        # 0:T2, 1:ADC, 2:High B, 3: TZ mask, 4: PZ mask

        # clip ADC
        images[1] = images[1].clamp(800, 2400)
        # normalize images
        images[:3] /= images[:3].amax(dim=(1,2,3), keepdim=True)

        results = {'img': images, 'mask': mask, 'path': fd_path}

        size = 128
        transforms = Compose([
            RandomFlip(),
            # RandomScale(),
            RandomCrop(size, size),
        ])

        if self.split == 'train':
            results = transforms(results)
        else:
            # center crop
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    _, _, h, w = v.shape
                    results[k] = v[:, :, (h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2]

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.to(torch.float)

        return results
