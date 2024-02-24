from dataloader import Dataloader_3D
from models import nnUNet25D

import time
import argparse
from datetime import datetime
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from vlkit.lrscheduler import CosineScheduler
from vlkit import set_random_seed
from vlkit import get_logger


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser=argparse.ArgumentParser(description='Detection Project')
    parser.add_argument('--comments',default="Nop",type=str,help='Add whatever you want to for future reference')
    parser.add_argument('--epochs',default=30,type=int,help='number of total epochs to run')
    parser.add_argument('--comment', default='', type=str)
    # parser.add_argument('--input_size',default=128,type=int,help='input size of the HW dimension of the image')
    parser.add_argument('--lr',default=1e-2,type=float,help='learning rate')
    parser.add_argument('--pfile_path', default='cv_ids_withNeg.p', type=str,help='relative path of each case, with grouping for cross-validation')
    parser.add_argument('--data_path',default='datasets/recentered_corrected/', type=str,help='absolute path of the whole dataset')
    
    parser.add_argument('--work_dir',default='',type=str,help='dataset using')
    
    parser.add_argument('--batch_size',default=4, type=int,help='Overall size of each batch')
    parser.add_argument('--label_set',default=2, type=float, help='0: GS=6, 1: GS>=6, 2: GS>=7, 3: TP(GS=6), 4: TP(GS>=6), 5: TP(GS>=7), 6: FN, 7: FP')
    parser.add_argument('--pe_dim',default=32, type=int,help='The dimension of vector of positional embedding')
    
    parser.add_argument('--is_focalloss_bceonly',default=False,help='Will we only use BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--is_focalloss_bceweighted',default=True,help='Will we weighted while using BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--focal_alpha',default=0.75,type=float,help='The parameter Alpha in the formula of Focalloss')                                                                          
    parser.add_argument('--focal_gamma',default=2.0,type=float,help='The parameter Gamma in the formula of Focalloss')                                                                           
    parser.add_argument('--bce_weight',default=30.0,type=float,help='Foreground weight when calculating BCE or Focalloss')
    parser.add_argument('--pretrained',default="", type=str, help='path for pretrained model')
    return parser.parse_args()

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


def validate(network, dataloader, args):
    network.eval()
    c=0
    loss=0

    for batch, data in enumerate(dataloader):
        img, mask = data['img'], data['mask']
        binary_zonal_mask = data['binary_zonal_mask']

        img=img.to(device)
        mask=mask.to(device)
        binary_zonal_mask = binary_zonal_mask.to(device)

        # 0601 Haoxin edited for zonal distance map
        img = torch.cat((img, binary_zonal_mask), axis=1)

        logits, pred = network(img)
        loss += focal_loss(args, logits, mask)
    return loss / len(dataloader)


def train(args):
    now=datetime.now()
    current_time=now.strftime("%m-%d-%Y_%H:%M:%S")

    os.makedirs(args.work_dir, exist_ok=True)
    num_channels = 5

    # We do 5-fold cross-validation
    for val_idx in [0, 1, 2, 3, 4]:

        # Select network you want to use.
        network=nnUNet25D(in_channels=num_channels, out_channels=1)
        network=network.to(device)

        optimizer=torch.optim.Adam(network.parameters(), lr=args.lr)            

        train_dataset=Dataloader_3D(args, split='train', val_idx=val_idx)
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            drop_last=True)
        val_dataloader=Dataloader_3D(args, split='val', val_idx=val_idx)
        val_loader=torch.utils.data.DataLoader(
            val_dataloader,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=None,
            drop_last=True)

        lrscheduler = CosineScheduler(
            max_iters=args.epochs * len(train_loader),
            warmup_iters=5 * len(train_loader),
            max_lr=args.lr,
            min_lr=1e-6
        )

        for epoch in range(args.epochs):
            network.train()
            for batch, data in enumerate(train_loader):
                img, mask = data['img'], data['mask']
                binary_zonal_mask = data['binary_zonal_mask']

                img=img.to(device)
                mask=mask.to(device)
                binary_zonal_mask = binary_zonal_mask.to(device)

                optimizer.zero_grad()

                # If you want to add a zonal positional encoding or not. Default: No
                img = torch.cat((img, binary_zonal_mask), axis=1)

                logits, pred = network(img) 
                loss = focal_loss(args, logits, mask)
                loss.backward()
                lr = lrscheduler.step()
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                
                if batch % 10 == 0:
                    logger.info(f"Fold [{val_idx}|5] epoch [{epoch}|{args.epochs}] iter [{batch}|{len(train_loader)}]: loss {loss.item():.3f}")
                    global_step =  len(train_loader) * epoch + batch
                    writer.add_scalar(f"fold-{val_idx}-train-loss", loss.item(), global_step=global_step)
                    writer.add_scalar(f"fold-{val_idx}-lr", lr, global_step=global_step)

            # validation
            with torch.no_grad():
                network.eval()
                # Validation loss 
                loss = validate(network,val_loader,args)
                msg="Val_idx:{}, Epoch:{}, Train-Loss:{:.4f},  Val-Loss:{:.4f}".format(val_idx, epoch,loss.item(), loss)
                logger.info(msg)
        torch.save(network.state_dict(), osp.join(args.work_dir, f'best_model_{val_idx}_final.pt'))

def main():
    set_random_seed(1115)    
    train(args)

if __name__=='__main__':
    main()
