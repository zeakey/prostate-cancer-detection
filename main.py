import torch, torchvision
from einops import rearrange
from dataloader import Dataloader_3D
from models import nnUNet25D, nnUNet3D
import argparse
from datetime import datetime
import numpy as np
import os, time
import pickle
import os.path as osp
import torch
import torch.nn.functional as F
import mmcv
from torch.utils.tensorboard import SummaryWriter
from vlkit.lrscheduler import CosineScheduler
from vlkit import set_random_seed
from vlkit import get_logger
from vlkit.image import normalize

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser=argparse.ArgumentParser(description='Detection Project')
    parser.add_argument('--epochs',default=30,type=int,help='number of total epochs to run')
    parser.add_argument('--images',default='t2,adc,highb,ktrans,kep,beta,pz_mask,tz_mask',type=str,help='Input images')
    parser.add_argument('--folds',default=5,type=int)
    parser.add_argument('--comment', default='', type=str)
    # parser.add_argument('--input_size',default=128,type=int,help='input size of the HW dimension of the image')
    parser.add_argument('--lr', default=1e-2, type=float,help='learning rate')
    parser.add_argument('--pfile_path', default='cv_ids_withNeg.p', type=str,help='relative path of each case, with grouping for cross-validation')
    parser.add_argument('--data',default='/home/kzhao/prostate-cancer-with-dce/dataset-with-dce', type=str,help='absolute path of the whole dataset')

    parser.add_argument('--work_dir', default='work_dirs/example', type=str)
    parser.add_argument('--workers',default=4, type=int)

    parser.add_argument('--batch_size',default=4, type=int,help='Overall size of each batch')
    parser.add_argument('--label_set',default=2, type=float, help='0: GS=6, 1: GS>=6, 2: GS>=7, 3: TP(GS=6), 4: TP(GS>=6), 5: TP(GS>=7), 6: FN, 7: FP')
    parser.add_argument('--pe_dim',default=32, type=int,help='The dimension of vector of positional embedding')
    
    parser.add_argument('--is_focalloss_bceonly',default=False,help='Will we only use BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--is_focalloss_bceweighted',default=True,help='Will we weighted while using BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--focal_alpha',default=0.75,type=float,help='The parameter Alpha in the formula of Focalloss')                                                                          
    parser.add_argument('--focal_gamma',default=2.0,type=float,help='The parameter Gamma in the formula of Focalloss')                                                                           
    parser.add_argument('--bce_weight',default=30.0,type=float,help='Foreground weight when calculating BCE or Focalloss')
    parser.add_argument('--pretrained',default="", type=str, help='path for pretrained model')
    args = parser.parse_args()
    args.images = args.images.split(',')
    return args

args = parse_args()

writer = SummaryWriter(
    log_dir=osp.join(args.work_dir, 'tensorboard'),
    comment=args.comment,
)

logger = get_logger(name='prostate', log_file=osp.join(args.work_dir, 'log.txt'))


def focal_loss(args, logits, target):
    alpha = args.focal_alpha
    gamma = args.focal_gamma
    bce_weight = args.bce_weight

    weight_map = torch.ones(target.shape)
    weight_map[target==1]=bce_weight
    weight_map = weight_map.to(device)
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    
    if args.is_focalloss_bceonly:
        if args.is_focalloss_bceweighted:
            focal_loss=torch.mean(bce_loss*weight_map)
        else:
            focal_loss=torch.mean(bce_loss)
    else:
        if args.is_focalloss_bceweighted:
            pt = torch.exp(-bce_loss)
            focal_loss = torch.mean(alpha*((1-pt)**gamma)*bce_loss*weight_map)
        else:
            pt = torch.exp(-bce_loss)
            focal_loss = torch.mean(alpha*((1-pt)**gamma)*bce_loss)

    return focal_loss



def train(args):
    now=datetime.now()
    current_time=now.strftime("%m-%d-%Y_%H:%M:%S")

    os.makedirs(args.work_dir, exist_ok=True)


    tic = time.time()
    # We do 5-fold cross-validation
    for fold in range(args.folds):

        # Select network you want to use.
        network = nnUNet25D(in_channels=len(args.images), out_channels=1)
        network = network.to(device)

        # optimizer=torch.optim.Adam(network.parameters(), lr=args.lr)
        optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        train_dataset=Dataloader_3D(data_root=args.data, split='train', fold=fold, folds=args.folds)
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=True)
        val_dataset=Dataloader_3D(data_root=args.data, split='val', fold=fold, folds=args.folds)
        val_loader=torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=True)

        lrscheduler = CosineScheduler(
            max_iters=args.epochs * len(train_loader),
            warmup_iters=5 * len(train_loader),
            max_lr=args.lr,
            min_lr=1e-4
        )

        for epoch in range(args.epochs):
            network.train()
            for batch, data in enumerate(train_loader):
                mask = data['mask']
                img = torch.cat([data[im] for im in args.images], dim=1)

                img=img.to(device)
                mask=mask.to(device)

                optimizer.zero_grad()

                logits, pred = network(img)
                loss = focal_loss(args, logits, mask.to(bool).to(torch.float))
                loss.backward()
                lr = lrscheduler.step()
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()

                if batch % 10 == 0 or batch == len(train_loader) - 1:
                    # find a representative image and slice
                    img_idx = mask.sum(dim=(2,3,4)).argmax().item()
                    slice_idx = mask[img_idx].sum(dim=(2, 3)).argmax().item()
                    case_id = data['case_id'][img_idx]
                    # extract slice mask
                    pz_mask = img[img_idx, -2, slice_idx,]
                    tz_mask = img[img_idx, -1, slice_idx,]
                    lesion_mask = mask[img_idx, 0, slice_idx].to(bool)
                    pred = pred[img_idx, 0, slice_idx]
                    pz_mask = rearrange(pz_mask, 'h w ->  1 1 h w').repeat(1, 3, 1, 1)
                    tz_mask = rearrange(tz_mask, 'h w ->  1 1 h w').repeat(1, 3, 1, 1)
                    lesion_mask = rearrange(lesion_mask, 'h w ->  1 1 h w').repeat(1, 3, 1, 1)
                    # image = t2 adc dwi
                    image = img[img_idx, :-2, slice_idx]
                    image = rearrange(image, 'n h w -> n 1 h w').repeat(1, 3, 1, 1)
                    red = torch.zeros_like(pz_mask)
                    green = torch.zeros_like(pz_mask)
                    blue = torch.zeros_like(pz_mask)
                    red[:, 2].fill_(1)
                    green[:, 1].fill_(1)
                    blue[:, 0].fill_(1)
                    alpha = 0.2
                    image = torch.cat((
                        image,
                        alpha * (red * pz_mask + blue * tz_mask) + (1 - alpha) * image,
                        alpha * (green * lesion_mask) + (1 - alpha) * image,
                        alpha * (blue * pred) + (1 - alpha) * image,
                        rearrange(pred, 'h w -> 1 1 h w').repeat(3, 3, 1, 1)
                    ))
                    total_iters = args.folds * args.epochs * len(train_loader)
                    passed_iters = fold * args.epochs * len(train_loader) + epoch * len(train_loader) + batch + 1
                    eta = (total_iters / passed_iters - 1) * (time.time() - tic)
                    eta = eta / 3600
                    logger.info(f"Fold [{fold}|{args.folds}] epoch [{epoch}|{args.epochs}] iter [{batch}|{len(train_loader)}] ({passed_iters}/{total_iters}): lr {lr: .3e}, loss {loss.item():.3f}. ETA={eta:.1f} hours.")
                    #
                    grid = torchvision.utils.make_grid(image, normalize=True, nrow=len(args.images)-2)
                    global_step = len(train_loader) * epoch + batch
                    writer.add_scalar(f"fold-{fold}-train-loss", loss.item(), global_step=global_step)
                    writer.add_scalar(f"fold-{fold}-lr", lr, global_step=global_step)
                    writer.add_image(f'{fold}-samples', grid, global_step=global_step)
                    #
                    
                    grid = normalize(rearrange(grid, 'c h w -> h w c').cpu().numpy(), 0, 255).astype(np.uint8)
                    mmcv.imwrite(grid, osp.join(args.work_dir, 'images', f'fold{fold}-epoc{epoch}-iter{batch}-{case_id}.png'))
            torch.save(network.state_dict(), osp.join(args.work_dir, f'best_model_{fold}_final.pt'))
            # validation
            with torch.no_grad():
                network.eval()
                loss = 0
                for batch, data in enumerate(val_loader):
                    mask = data['mask']
                    img = torch.cat([data[im] for im in args.images], dim=1)

                    img=img.to(device)
                    mask = mask.to(device)

                    logits, pred = network(img)
                    loss += focal_loss(args, logits, mask.to(bool).to(torch.float))

                    if (epoch+1) % 5 == 0 or (epoch+1) == args.epochs:
                        save_dir = osp.join(args.work_dir, 'inference', f'epoch-{epoch}')
                        os.makedirs(save_dir, exist_ok=True)
                        for i, case_id in enumerate(data['case_id']):
                            pickle.dump(pred[i, 0].to(torch.device('cpu')), open(osp.join(save_dir, case_id+"_pred_{:.4f}.p".format(0.00001)), 'wb'))
                            pickle.dump(mask[i, 0].to(torch.device('cpu')), open(osp.join(save_dir, case_id+"_mask_{:.4f}.p".format(0.00001)), 'wb'))

                loss /= len(val_loader)
                logger.info(f"Fold-{fold} epoch-{epoch} val loss: {loss.item():.4e}")
                writer.add_scalar(f'Fold-{fold}-val-loss', loss.item(), epoch)

def main():
    set_random_seed(1115)    
    train(args)

if __name__=='__main__':
    main()
