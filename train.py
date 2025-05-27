from argparse import ArgumentParser

import torch
import torch.utils.data as Data
from data.data import DataLoaderX, GrokCV_NUAA_SIRST, GrokCV_NUDT_SIRST, GrokCV_IRSTD_1k, GrokCV_SIRST_Aug

from model.model import DS_TransNet
from model.model_utils.weight_init import weight_init_normal, weights_init_kaiming
from model.model_utils.optimizer_set import optimizer_set_adam, optimizer_set_adagrad, optimizer_set_sgd
from model.model_utils.lr_scheduler import lr_scheduler_WarmDecayLR, lr_scheduler_WarmConstantDecayLR, lr_scheduler_MultiStepLR, lr_scheduler_CosineAnnealingLR_With_GradualWarmup
from model.model_utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, PD_FA


from utils.tools import denormalize, set_seed, init_env

from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
import numpy as np
from tqdm import tqdm



def parse_args():

    parser = ArgumentParser(description='Implement of DS_TransNet')
    
    # log
    parser.add_argument('--log_root', type=str, default="logs", help='log dir')
    parser.add_argument('--exp_name', type=str, default="DS_TransNet", help='experiment name')
    parser.add_argument('--phase_name', type=str, default="train", help='phase name')
    parser.add_argument('--log_name', type=str, default="log.log", help='log name')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')

    # environment
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--seed", type=int, default=3407, help="Torch seed 3407 is all you need")


    # scheduler
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')

    parser.add_argument('--lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='minimum learning rate')

    parser.add_argument('--warmdecaylr', type=dict, default={'warm_up_epochs': 0}, help="lr_scheduler_WarmDecayLR")
    parser.add_argument('--warmconstantdecaylr', type=dict, default={'warm_up_epochs': 0, 'constant_epochs': 210}, help="lr_scheduler_WarmConstantDecayLR")
    parser.add_argument("--multisteplr", type=dict, default={'milestones': [150, 200, 260], 'gamma': 0.5}, help="lr_scheduler_MultiStepLR")
    parser.add_argument("--cosineAnnealinglr", type=dict, default={'last_epoch': -1}, help="lr_scheduler_CosineAnnealingLR")

    args = parser.parse_args()
    
    return args


class Trainer(object):

    def __init__(self, args):
        self.args = args


        # self.trainset = GrokCV_NUAA_SIRST(mode='train')
        # self.valset = GrokCV_NUAA_SIRST(mode='test')
        # self.trainset = GrokCV_NUDT_SIRST(mode='train')
        # self.valset = GrokCV_NUDT_SIRST(mode='test')
        # self.trainset = GrokCV_IRSTD_1k(mode='train')
        # self.valset = GrokCV_IRSTD_1k(mode='test')
        self.trainset = GrokCV_SIRST_Aug(mode='train')
        self.valset = GrokCV_SIRST_Aug(mode='test')
        
        # self.train_data_loader = Data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)
        # self.val_data_loader = Data.DataLoader(self.valset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)
        self.train_data_loader = DataLoaderX(self.trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)
        self.val_data_loader = DataLoaderX(self.valset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

        # dir
        self.log_dir = os.path.join(args.log_root, args.exp_name, args.phase_name, time.strftime('%Y%m%d%H%M%S', time.localtime()) )
        if not os.path.exists( self.log_dir ):
            os.makedirs( self.log_dir )
        # log
        self.log_file = os.path.join( self.log_dir, args.log_name )
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        # tensorboard
        self.writer = SummaryWriter(self.log_dir)

        self.writer.add_text(self.log_dir, 'Args:%s' % args)
        # for i, (data, label) in enumerate( tqdm(self.train_data_loader, desc='Export trainset to TensorBoard      ', position=0, leave=True) ):
        #     self.writer.add_images('img/original', denormalize(data, self.trainset.mean, self.trainset.std), i, dataformats='NCHW')
        #     self.writer.add_images('img/processed', data, i, dataformats='NCHW')
        #     self.writer.add_images('label', label, i, dataformats='NCHW')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = DS_TransNet(mode='train')

        # weight init 
        # self.net.apply(weight_init_normal)
        self.net.apply(weights_init_kaiming)
        self.net.train()# will change in training stage and validation stage
        self.net = self.net.to(self.device)#all parameter including extra parameter
        self.net.model = self.net.model.to(self.device)# only model parameter

        # optimizer set
        # self.optimizer = optimizer_set_adagrad(self.net.parameters(), args.lr)
        self.optimizer = optimizer_set_adam(self.net.parameters(), args.lr)
        # self.optimizer = optimizer_set_sgd(self.net.parameters(), args.lr)
        
        # lr_scheduler
        # self.scheduler = lr_scheduler_WarmDecayLR(self.optimizer, args.lr, args.epochs, args.warmdecaylr, args.min_lr)
        # self.scheduler = lr_scheduler_WarmConstantDecayLR(self.optimizer, args.lr, args.epochs, args.warmconstantdecaylr, args.min_lr)
        # self.scheduler = lr_scheduler_MultiStepLR(self.optimizer, args.multisteplr)
        self.scheduler = lr_scheduler_CosineAnnealingLR_With_GradualWarmup(self.optimizer, args.epochs, args.min_lr, args.cosineAnnealinglr)

        # loss
        self.net.cal_loss = self.net.cal_loss.to(self.device)# only loss function parameter


        # metric
        self.miou_metric = SigmoidMetric(score_thresh=0.5)
        self.nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5, do_sigmoid=False)
        self.best_miou = 0
        self.best_nIoU = 0


    def training(self, epoch):
        self.net.train()
        losses = []
        tbar = tqdm(self.train_data_loader, desc='Iteration...', position=1, leave=True)
        for i, (data, label) in enumerate( tbar ):
            # data  = data[:, 1:2, :, :] # for single channel only
            data = data.to(self.device)
            label = label.to(self.device)
            
            output = self.net(data)
            loss = self.net.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            # You can't print too frequently, as it will bring down GPU utilization
            # tbar.set_description('Iteration...  Epoch:%3d, lr:%f, train loss:%f'% (epoch, self.optimizer.param_groups[0]['lr'], np.mean(losses)))
        
        self.scheduler.step()

        self.writer.add_scalar('Losses/train_loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], epoch)
        self.logger.info('Epoch: %d, train_loss: %.4f, lr: %.6f' % (epoch, np.mean(losses), self.optimizer.param_groups[0]['lr']))


    def validation(self, epoch):
        self.net.eval()
        self.miou_metric.reset()
        self.nIoU_metric.reset()
        losses = []
        with torch.no_grad():
            tbar = tqdm(self.val_data_loader, desc='Validation...', position=2, leave=True)
            for i, (data, label) in enumerate( tbar ):
                # data  = data[:, 1:2, :, :] # for single channel only
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = self.net.loss(output, label)

                losses.append(loss.item())

                self.miou_metric.update(output[5], label)
                self.nIoU_metric.update(output[5], label)
                # print('output max value: %.2f, min value: %.2f' % (torch.max(output[5]), torch.min(output[5])) )
                _, miou = self.miou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                tbar.set_description('Validation...  Epoch:%3d, eval loss:%f, mIoU:%f, nIoU:%f'% (epoch, np.mean(losses), miou, nIoU))

        _, miou = self.miou_metric.get()
        _, nIoU = self.nIoU_metric.get()

        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(self.net.mode.state_dict(), os.path.join(self.log_dir, 'best_miou.pth'))
        if nIoU > self.best_nIoU:
            self.best_nIoU = nIoU
            torch.save(self.net.model.state_dict(), os.path.join(self.log_dir, 'best_nIoU.pth'))
        
        self.writer.add_scalar('Losses/val_loss', np.mean(losses), epoch)
        self.writer.add_scalar('Eval/mIoU', miou, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/best_mIoU', self.best_miou, epoch)
        self.writer.add_scalar('Best/best_nIoU', self.best_nIoU, epoch)
        self.logger.info('Epoch: %d, val_loss: %.4f, mIoU: %.4f, best_mIoU: %.4f, nIoU: %.4f, best_nIoU: %.4f' % (epoch, np.mean(losses), miou, self.best_miou, nIoU, self.best_nIoU))


if __name__ == '__main__':

    args = parse_args()

    init_env(args.gpu_ids)

    # set_seed(args.seed)

    trainer = Trainer(args)

    for epoch in  tqdm(range(args.epochs), desc='Epoch...', position=0, leave=True) :
        trainer.training(epoch)
        if epoch<300:
            if epoch % 5 == 0:
                trainer.validation(epoch)
        else:
            trainer.validation(epoch)
