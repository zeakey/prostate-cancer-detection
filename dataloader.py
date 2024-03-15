import torch.utils.data
import os, sys
import os.path as osp
from glob import glob
import torch, torchvision
from scipy.io import loadmat
from torchvision.transforms import Compose
from vlkit.transforms import center_crop
import numpy as np
import pickle


blurer = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

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


class CenterCrop:
    def __init__(self, size) -> None:
        self.size = size
        assert size % 2 == 0

    def __call__(self, results):
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                h, w = v.shape[-2:]
                assert min(h, w) >= self.size, f"image shape ({h}x{w}) v.s. size {self.size}"
                results[k] = v[:, :, (h - self.size) // 2 : (h + self.size) // 2, (w - self.size) // 2 : (w + self.size) // 2, ]
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
    def __init__(self, data_root, split, fold, folds=5):

        assert fold < folds
        self.cases = np.array([i for i in glob(f"{data_root}/*") if osp.isdir(i)])
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

        self.slices = 20

        self.fold = fold
        self.folds = folds
        self.split = split

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        case = self.case_list[item]
        t2_adc_highb = torch.tensor(np.load(osp.join(case, 't2-adc-highb.npy')))
        dce = torch.tensor(np.load(osp.join(case, 'ktrans-beta-kep.npy')))
        lesion_mask = torch.tensor(np.load(osp.join(case, 'lesion_mask.npy')))
        zonal_mask = torch.tensor(np.load(osp.join(case, 'zonal_mask.npy')))
        # mask dce
        dce = dce * (zonal_mask != 0)
        #
        t2_adc_highb[:, 1,] = t2_adc_highb[:, 1,].clamp(800, 2400)
        t2_adc_highb /= t2_adc_highb.amax(dim=(1,2,3), keepdim=True)

        images = torch.cat((
            t2_adc_highb,
            dce,
            zonal_mask==1,
            zonal_mask==2), dim=0)

        # sanity check
        h, w = images.shape[-2:]
        slices = images.shape[1]
        assert lesion_mask.shape[-2:] == (h, w)
        assert slices >= 20
        images = images[:, slices // 2 - 10: slices // 2 + 10]
        lesion_mask = lesion_mask[:, slices // 2 - 10: slices // 2 + 10]
        slices = 20
        t2, adc, highb, ktrans, beta, kep, pz_mask, tz_mask = images.split(1, dim=0)
        results = dict(
            t2=t2,
            adc=adc,
            highb=highb,
            ktrans=ktrans,
            beta=beta,
            kep=kep,
            pz_mask=pz_mask,
            tz_mask=tz_mask,
            mask=lesion_mask,
            case_id=osp.splitext(case.split(os.sep)[-1])[0]
        )

        size = 128
        transforms_train = Compose([
            CenterCrop(size=150),
            RandomFlip(),
            # RandomScale(),
            RandomCrop(size, size),
        ])
        transforms_test = Compose([
            CenterCrop(size=size),
        ])

        if self.split == 'train':
            results = transforms_train(results)
        else:
            results = transforms_test(results)

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.to(torch.float)

        return results
