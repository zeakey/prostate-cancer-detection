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
        self.cases = np.array([i for i in glob(f"{data_root}/*.mat")])
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
        data = loadmat(case)
        lesion_mask = torch.tensor(data['lesion_mask']).unsqueeze(dim=0).to(bool).to(torch.float)
        zonal_mask = torch.tensor(data['zonal_mask']).unsqueeze(dim=0)
        t2 = torch.tensor(data['t2']).unsqueeze(dim=0)
        adc = torch.tensor(data['adc']).unsqueeze(dim=0)
        adc = adc.clamp(800, 2400)
        highb = torch.tensor(data['highb']).unsqueeze(dim=0)
        # dce
        try:
            ktrans = torch.tensor(data['ktrans']).unsqueeze(dim=0)
        except:
            print('??')
        kep = torch.tensor(data['kep']).unsqueeze(dim=0)
        beta = torch.tensor(data['beta']).unsqueeze(dim=0)

        images = torch.cat((
            t2, adc, highb,
            # ktrans, kep, beta,
            zonal_mask==1,
            zonal_mask==2), dim=0)

        # images_path = os.path.join(fd_path, "full_stack_data.p")
        # images = pickle.load(open(images_path, 'rb'))

        # mask_path = os.path.join(fd_path, "full_stack_mask_label_set_{}.p".format(self.label_set)) 
        # mask = pickle.load(open(mask_path, 'rb'))

        # images = torch.tensor(images[:, :, self.crop_start:self.crop_end, self.crop_start:self.crop_end], dtype=torch.float)
        # mask = torch.tensor(mask[:, :, self.crop_start:self.crop_end, self.crop_start:self.crop_end], dtype=torch.uint8)

        # sanity check
        h, w = images.shape[-2:]
        slices = images.shape[1]
        assert lesion_mask.shape[-2:] == (h, w)
        assert slices >= 20
        images = images[:, slices // 2 - 10: slices // 2 + 10]
        lesion_mask = lesion_mask[:, slices // 2 - 10: slices // 2 + 10]
        slices = 20

        # tricks
        # noise = torch.rand(mask.shape) / 10
        # if mask.max() > 0 and np.random.uniform() >= 0.7:
        #     noise = noise * (mask == 0) + \
        #                  noise * (mask == 128) * np.random.uniform(1.1, 2) + \
        #                  noise * (mask == 255) * np.random.uniform(1.1, 4)
        #     # noise[mask == 128] += 0.05
        #     # noise[mask == 255] += 0.01
        # noise = blurer(noise)
        # torchvision.utils.save_image(noise.transpose(0, 1), 'noise.png', normalize=True)
        # images[0] = images[0] + noise

        results = {'img': images, 'mask': lesion_mask, "case_id": osp.splitext(case.split(os.sep)[-1])[0]}

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
        results['img'][:-2] /= results['img'][:-2].amax(dim=(1,2,3), keepdim=True)

        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.to(torch.float)

        return results
