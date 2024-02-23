from dataloader import Dataloader_3D
from models import nnUNet25D

import time, wandb
import argparse
from datetime import datetime
import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
from vlkit.lrscheduler import CosineScheduler
from vlkit import set_random_seed


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser=argparse.ArgumentParser(description='Detection Project')
    parser.add_argument('--comments',default="Nop",type=str,help='Add whatever you want to for future reference')
    parser.add_argument('--epochs',default=30,type=int,help='number of total epochs to run')
    parser.add_argument('--input_size',default=128,type=int,help='input size of the HW dimension of the image')
    parser.add_argument('--lr',default=1e-2,type=float,help='learning rate')
    parser.add_argument('--pfile_path', default='cv_ids_withNeg.p', type=str,help='relative path of each case, with grouping for cross-validation')
    parser.add_argument('--data_path',default='datasets/recentered_corrected/', type=str,help='absolute path of the whole dataset')
    
    parser.add_argument('--work_dir',default='',type=str,help='dataset using')
    parser.add_argument('--try_id',default='0',type=str,help='id of try')
    parser.add_argument('--is_regularization',default=False,help='Whether we add regularization in optimizer or not',action='store_true')
    parser.add_argument('--reg_val',default=1e-5,type=float,help='How much regularization we want to add')
    parser.add_argument('--batch_size',default=4,type=int,help='Overall size of each batch')   
    parser.add_argument('--label_set',default=2,type=float,help='0: GS=6, 1: GS>=6, 2: GS>=7, 3: TP(GS=6), 4: TP(GS>=6), 5: TP(GS>=7), 6: FN, 7: FP')
    parser.add_argument('--pe_dim',default=32, type=int,help='The dimension of vector of positional embedding')
    
    parser.add_argument('--is_focalloss_bceonly',default=False,help='Will we only use BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--is_focalloss_bceweighted',default=True,help='Will we weighted while using BCE loss while applyFocal Loss',action='store_true')
    parser.add_argument('--focal_alpha',default=0.75,type=float,help='The parameter Alpha in the formula of Focalloss')                                                                          
    parser.add_argument('--focal_gamma',default=2.0,type=float,help='The parameter Gamma in the formula of Focalloss')                                                                           
    parser.add_argument('--bce_weight',default=30.0,type=float,help='Foreground weight when calculating BCE or Focalloss')
    
    parser.add_argument('--exclude_t2',default=False,help='Will we exclude T2 Images',action='store_true')
    parser.add_argument('--exclude_adc',default=False,help='Will we exclude ADC Images',action='store_true')
    parser.add_argument('--exclude_dwi',default=False,help='Will we exclude DWI Images',action='store_true')   
    
    parser.add_argument('--include_zonal_pe_encoding',default=False, help='if we use the zonal distance map instead of all-one pure zonal mask ?',action='store_true')
    parser.add_argument('--include_zonal_binary_mask',default=True,help='Will we include zonal masks',action='store_true')
    parser.add_argument('--is_background_pe_encoded',default=False,help='If we do PE for background voxels',action='store_true')
    
    parser.add_argument('--is_load_pretrained_model',default=False,help='If we load pretrained model or not',action='store_true')
    parser.add_argument('--pretrained_model_path',default="", type=str, help='path for pretrained model')
    parser.add_argument('--server_name',default='stellaris', type=str, help='Which server are we using? This affects the abs path of data')
    parser.add_argument('--mode', default='nnUNet', type=str, help='mode name to be used')
    parser.add_argument("--early_stopping_epoch", default=5, type=int, help='how many epoches without update then we terminate')
    parser.add_argument("--is_early_stopping", default=True, action='store_true', help='how many epoches without update then we terminate')
    return parser.parse_args()

args = parse_args()

# start a new wandb run to track this script
wandb.init(
    dir=osp.join(args.work_dir, 'wandb'),
    name=args.work_dir.split(os.sep)[-1],
    tags=['lr1e-3', 'cosine-lr', 'baseline'],
    project="prostate-cancer-detection",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    }
)

# used for store a log for future reference
def print_log(print_string,log): 
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


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
        c+=1
        img, mask = data['img'], data['mask']
        pe_encoded_zonal_mask, binary_zonal_mask = data['pe_encoded_zonal_mask'], data['binary_zonal_mask']

        img=img.to(device)
        mask=mask.to(device)
        pe_encoded_zonal_mask=pe_encoded_zonal_mask.to(device)
        binary_zonal_mask = binary_zonal_mask.to(device)

        # 0601 Haoxin edited for zonal distance map
        if args.include_zonal_binary_mask:
            img = torch.cat((img, binary_zonal_mask), axis=1)
        if args.include_zonal_pe_encoding:
            img = torch.cat((img, pe_encoded_zonal_mask), axis=1)

        logits, pred = network(img)

        loss_grad = 0
        loss_grad += focal_loss(args, logits, mask)

        loss += loss_grad.item()

    return loss/c


def train(args):
    now=datetime.now()
    current_time=now.strftime("%m-%d-%Y_%H:%M:%S")

    os.makedirs(args.work_dir, exist_ok=True)

    log=open(args.work_dir+"model_training_log_id-{}_t-{}.txt".format(args.try_id,current_time),'w')
    state={k:v for k,v in args._get_kwargs()}

    # generate logs e.g. {'alpha': 1.0, 'batch_size': 4, 'belta': 1.0, ...
    print_log(state,log)  

    if args.include_zonal_pe_encoding and args.include_zonal_binary_mask:
        num_channels = args.pe_dim + 3 + 2
    elif args.include_zonal_pe_encoding and not args.include_zonal_binary_mask:
        num_channels = args.pe_dim + 3 
    elif not args.include_zonal_pe_encoding and args.include_zonal_binary_mask:
        num_channels = 3 + 2
    else:
        num_channels = 3

    # We do 5-fold cross-validation
    for val_idx in [0, 1, 2, 3, 4]:

        # Select network you want to use.
        network=nnUNet25D(in_channels=num_channels, out_channels=1)

        # If you want to load a model with a given weight.
        if args.is_load_pretrained_model:
            curr_model_path = osp.join(args.pretrained_model_path, "best_model_{}.pt".format(val_idx))
            network.load_state_dict(torch.load(curr_model_path))

        network=network.to(device)
        print("Load Network Successfully!")

        if args.is_regularization:
            optimizer=torch.optim.Adam(network.parameters(),lr=args.lr,weight_decay=args.reg_val)
        else:
            optimizer=torch.optim.Adam(network.parameters(),lr=args.lr)            

        train_dataset=Dataloader_3D(args, split='train', val_idx=val_idx)
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
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
        early_stopping_cnt = 0
        for epoch in range(args.epochs):
            network.train()

            loss_item = 0
            c = 0
            start_time = time.time()
            for batch, data in enumerate(train_loader):
                img, mask = data['img'], data['mask']
                pe_encoded_zonal_mask, binary_zonal_mask = data['pe_encoded_zonal_mask'], data['binary_zonal_mask']

                img=img.to(device)
                mask=mask.to(device)
                pe_encoded_zonal_mask=pe_encoded_zonal_mask.to(device)
                binary_zonal_mask = binary_zonal_mask.to(device)

                optimizer.zero_grad()    

                # If you want to add a zonal positional encoding or not. Default: No
                if args.include_zonal_binary_mask:
                    img = torch.cat((img, binary_zonal_mask), axis=1)
                if args.include_zonal_pe_encoding:
                    img = torch.cat((img, pe_encoded_zonal_mask), axis=1)             

                logits, pred = network(img) 
                loss = focal_loss(args, logits, mask)

                c+=1

                loss.backward()

                lr = lrscheduler.step()
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                wandb.log(
                    data = {
                        f"fold-{val_idx}-train-loss": loss.item(),
                        f"fold-{val_idx}-lr": lr
                    },
                    step = len(train_loader) * epoch + batch
                )

            # validation
            with torch.no_grad():
                network.eval()
                # Validation loss 
                loss = validate(network,val_loader,args)
                wandb.log(
                    data = {f"fold-{val_idx}-val-loss": loss},
                    step = len(train_loader) * (epoch + 1),
                )
                msg="Val_idx:{}, Epoch:{}, Train-Loss:{:.4f},  Val-Loss:{:.4f}".format(val_idx, epoch,loss_item/c, loss)
                print_log(msg,log)
            end_time = time.time()
            print("@@@ Time Cost: {}".format(end_time - start_time))
        torch.save(network.state_dict(), osp.join(args.work_dir, f'best_model_{val_idx}_final.pt'))

def main():
    set_random_seed(1115)    
    train(args)

if __name__=='__main__':
    main()
