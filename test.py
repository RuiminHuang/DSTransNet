from argparse import ArgumentParser

import torch
import torch.utils.data as Data
from data.data import DataLoaderX, GrokCV_NUAA_SIRST, GrokCV_NUDT_SIRST, GrokCV_IRSTD_1k, GrokCV_SIRST_Aug
from model.model import DS_TransNet
from model.model_utils.weight_init import weight_init_normal
from model.model_utils.optimizer_set import optimizer_set_adam, optimizer_set_adagrad, optimizer_set_sgd
from model.model_utils.lr_scheduler import lr_scheduler_WarmDecayLR, lr_scheduler_WarmConstantDecayLR, lr_scheduler_MultiStepLR
from model.model_utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, SeRankDet_PD_FA, PD_FA

import torch.nn.functional as F

from utils.tools import denormalize, set_seed, init_env

from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from tqdm import tqdm
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from torchvision.utils import save_image

def parse_args():

    parser = ArgumentParser(description='Implement of DS_TransNet')
    
    # log
    parser.add_argument('--log_root', type=str, default="logs", help='log dir')
    parser.add_argument('--exp_name', type=str, default="DS_TransNet", help='experiment name')
    parser.add_argument('--phase_name', type=str, default="test", help='phase name')
    parser.add_argument('--log_name', type=str, default="log.log", help='log name')

    # testing parameters
    # parser.add_argument('--weight_path', type=str, default="./logs/DS_TransNet/train/20250318003841/best_nIoU.pth", help='weight for testing')# NUAA_SIRST
    # parser.add_argument('--weight_path', type=str, default="./logs/DS_TransNet/train/20250117023840/best_miou.pth", help='weight for testing')# NUDT
    # parser.add_argument('--weight_path', type=str, default="./logs/DS_TransNet/train/20250312104502/best_nIoU.pth", help='weight for testing')# IRSTD_1k
    parser.add_argument('--weight_path', type=str, default="./logs/DS_TransNet/train/20250318095017/best_miou.pth", help='weight for testing')# SIRST-Aug

    parser.add_argument('--batch_size', type=int, default=1, help='batch_size for training')# must be 1 in inference stage & considering saveing image
    
    # environment
    parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--seed", type=int, default=3407, help="Torch seed 3407 is all you need")

    args = parser.parse_args()
    
    return args


class Tester(object):

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

        
        
        # self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4)
        # self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4)
        self.train_data_loader = DataLoaderX(self.trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=8)
        self.test_data_loader = DataLoaderX(self.valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=8)

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
        for i, (data, label) in enumerate( tqdm(self.test_data_loader, desc='Export valset to TensorBoard      ', position=0, leave=True) ):
            self.writer.add_images('img/original', denormalize(data, self.valset.mean, self.valset.std), i, dataformats='NCHW')
            self.writer.add_images('img/processed', data, i, dataformats='NCHW')
            self.writer.add_images('label/original', label, i, dataformats='NCHW')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model init

        self.net = DS_TransNet(mode='test')
        self.net.model.load_state_dict(torch.load(args.weight_path, weights_only=True))# method 1: load dict only
        # self.net.model = torch.load(args.weight_path)# method 2: load all model
        # self.net.load_state_dict(torch.load(args.weight_path)['state_dict'])# method 3: only used in SC_TransNet pre_trained model
        # print(torch.load(args.weight_path)['epoch'])# method 3: only used in SC_TransNet pre_trained model
        # print(torch.load(args.weight_path)['total_loss'])# method 3: only used in SC_TransNet pre_trained model
        
        self.net.eval()
        self.net = self.net.to(self.device)#all parameter including extra parameter
        self.net.model = self.net.model.to(self.device)# only model parameter

        # metric
        self.miou_metric = SigmoidMetric(score_thresh=0.5)
        self.nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5, do_sigmoid=False)# do_sigmoid=False, as the output has do sigmoid
        self.roc_metric = ROCMetric(nclass=1, bins=100, do_sigmoid=False)# do_sigmoid=False, as the output has do sigmoid
        # https://github.com/GrokCV/SeRankDet/blob/master/utils/metric.py (Also used in SCTransNet, DNA-Net and MTU-Net)
        self.SeRankDet_pd_fa_metric = SeRankDet_PD_FA(1, 100)
        # Modified based on BasicIRSTD, as a reference
        self.pd_fa_metric = PD_FA(1, 100)


    def testing(self):
        self.miou_metric.reset()
        self.nIoU_metric.reset()
        with torch.no_grad():
            tbar = tqdm(self.test_data_loader, desc='Testing...', position=0, leave=True)
            for i, (data, label) in enumerate( tbar ):
                # data  = data[:, 1:2, :, :] # for single channel only
                # self.writer.add_images('img/original_resize', denormalize(data, self.valset.mean, self.valset.std), i, dataformats='NCHW')
                # self.writer.add_images('img/processed_resize', data, i, dataformats='NCHW')
                # self.writer.add_images('label/original_resize', label, i, dataformats='NCHW')

                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)

                self.miou_metric.update(output, label)
                self.nIoU_metric.update(output, label)
                self.roc_metric.update(output, label)
                self.SeRankDet_pd_fa_metric.update(output, label, output[0, 0].shape)
                self.pd_fa_metric.update(output, label, output[0, 0].shape)

                # print('output max value: %.2f, min value: %.2f' % (torch.max(output), torch.min(output)) )

                # save output
                self.writer.add_images('result/output', output, i, dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output'), exist_ok=True)
                save_image(output, os.path.join(self.log_dir, 'output', 'output_%d.png' % (i+1) ))

                # save output_0_5_threshold
                output_0_5_threshold = output.clone()
                output_0_5_threshold[output_0_5_threshold > 0.5] = 1
                output_0_5_threshold[output_0_5_threshold <= 0.5] = 0
                self.writer.add_images('result/output_0_5_threshold', output_0_5_threshold, i, dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output_0_5_threshold'), exist_ok=True)
                save_image(output_0_5_threshold, os.path.join(self.log_dir, 'output_0_5_threshold', 'output_0_5_threshold_%d.png' % (i+1) ))

                # save output_0_5_threshold_visual
                overlay_color = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1).to(self.device)
                alpha = 0.6
                mask_3c = output_0_5_threshold.repeat(1, 3, 1, 1)
                output_0_5_threshold_visual = denormalize(data, self.valset.mean, self.valset.std) * (1 - alpha * mask_3c) + overlay_color * (alpha * mask_3c)
                self.writer.add_images('result/output_0_5_threshold_visual', output_0_5_threshold_visual, i, dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output_0_5_threshold_visual'), exist_ok=True)
                save_image(output_0_5_threshold_visual, os.path.join(self.log_dir, 'output_0_5_threshold_visual', 'output_0_5_threshold_visual_%d.png' % (i+1) ))

                _, miou = self.miou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = self.roc_metric.get()
                auc_value = auc(fp_rates, tp_rates)
                SeRankDet_Final_PD, SeRankDet_Final_PD_All, SeRankDet_Final_FA, SeRankDet_Final_FA_All = self.SeRankDet_pd_fa_metric.get()
                Final_PD, Final_PD_All, Final_FA, Final_FA_All = self.pd_fa_metric.get()

                tbar.set_description('Testing...  mIoU:%f, nIoU:%f, auc_value:%f, f1_score:%f'% (miou, nIoU, auc_value, f1_score))


        _, miou = self.miou_metric.get()
        _, nIoU = self.nIoU_metric.get()
        tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = self.roc_metric.get()
        auc_value = auc(fp_rates, tp_rates)
        SeRankDet_Final_PD, SeRankDet_Final_PD_All, SeRankDet_Final_FA, SeRankDet_Final_FA_All = self.SeRankDet_pd_fa_metric.get()
        Final_PD, Final_PD_All, Final_FA, Final_FA_All = self.pd_fa_metric.get()

        miou = miou * 1e+2
        nIoU = nIoU * 1e+2

        f1_score = f1_score * 1e+2
        f1_score_all = f1_score_all * 1e+2
        
        SeRankDet_Final_PD = SeRankDet_Final_PD * 1e+2
        SeRankDet_Final_PD_All = SeRankDet_Final_PD_All * 1e+2
        Final_PD = Final_PD * 1e+2
        Final_PD_All = Final_PD_All * 1e+2

        SeRankDet_Final_FA = SeRankDet_Final_FA * 1e+6
        SeRankDet_Final_FA_All = SeRankDet_Final_FA_All * 1e+6
        Final_FA = Final_FA * 1e+6
        Final_FA_All = Final_FA_All * 1e+6


        # ROC 曲线
        fig, ax = plt.subplots(figsize=(8, 6))  # 设置更宽松的画布尺寸
        ax.plot(fp_rates[1:-1], tp_rates[1:-1], color='#377eb8', lw=2, label='ROC curve (area = %0.2f)' % auc_value)  # 调整颜色
        ax.set_xlim([0.0, 3.10e-04])  # need to adjust
        ax.set_ylim([0.0, 1.05])  # correct
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')  # 调整字体
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold', pad=15)  # 加间距
        ax.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)  # 图例美化
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)  # 添加浅灰色网格线
        plt.tight_layout()
        # 保存到 TensorBoard
        self.writer.add_figure('ROC_curve', fig, 0)
        
        # Precision-Recall 曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall[1:-1], precision[1:-1], color='#984ea3', lw=2, label='Precision-Recall curve')  # 柔和紫色
        ax.set_xlim([0.0, 1.05]) # correct
        ax.set_ylim([0.0, 1.05]) # correct
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Characteristic', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        # 保存到 TensorBoard
        self.writer.add_figure('Precision_to_Recall_curve', fig, 0)


        self.logger.info('\r\n mIoU:\r\n%f\r\n nIoU:\r\n%f\r\n auc_value:\r\n%f\r\n f1_score:\r\n%f\r\n f1_score_all:\r\n%s\r\n tp_rates:\r\n%s\r\n fp_rates:\r\n%s\r\n recall:\r\n%s\r\n precision:\r\n%s\r\n SeRankDet_Final_PD:\r\n%s\r\n SeRankDet_Final_PD_All:\r\n%s\r\n SeRankDet_Final_FA:\r\n%s\r\n SeRankDet_Final_FA_All:\r\n%s\r\n Final_PD:\r\n%s\r\n Final_PD_All:\r\n%s\r\n Final_FA:\r\n%s\r\n Final_FA_All:\r\n%s\r\n'% (miou, nIoU, auc_value, f1_score, f1_score_all, tp_rates, fp_rates, recall, precision, SeRankDet_Final_PD, SeRankDet_Final_PD_All, SeRankDet_Final_FA, SeRankDet_Final_FA_All, Final_PD, Final_PD_All, Final_FA, Final_FA_All) )
        self.writer.add_text('mIoU', 'mIoU:%f'% (miou))
        self.writer.add_text('nIoU', 'nIoU:%f'% (nIoU))
        self.writer.add_text('auc_value', 'auc_value: %f'% (auc_value))
        self.writer.add_text('f1_score', 'f1_score:%f'% (f1_score))
        self.writer.add_text('f1_score_all', 'f1_score_all:%s'% (f1_score_all))
        self.writer.add_text('tp_rates', 'tp_rates:%s'% (tp_rates))
        self.writer.add_text('fp_rates', 'fp_rates:%s'% (fp_rates))
        self.writer.add_text('recall', 'recall:%s'% (recall))
        self.writer.add_text('precision', 'precision:%s'% (precision))
        self.writer.add_text('SeRankDet_Final_PD', 'SeRankDet_Final_PD:%s'% (SeRankDet_Final_PD))
        self.writer.add_text('SeRankDet_Final_PD_All', 'SeRankDet_Final_PD_All:%s'% (SeRankDet_Final_PD_All))
        self.writer.add_text('SeRankDet_Final_FA', 'SeRankDet_Final_FA:%s'% (SeRankDet_Final_FA))
        self.writer.add_text('SeRankDet_Final_FA_All', 'SeRankDet_Final_FA_All:%s'% (SeRankDet_Final_FA_All))
        self.writer.add_text('Final_PD', 'Final_PD:%s'% (Final_PD))
        self.writer.add_text('Final_PD_All', 'Final_PD_All:%s'% (Final_PD_All))
        self.writer.add_text('Final_FA', 'Final_FA:%s'% (Final_FA))
        self.writer.add_text('Final_FA_All', 'Final_FA_All:%s'% (Final_FA_All))

        self.writer.close()



if __name__ == '__main__':

    args = parse_args()

    init_env(args.gpu_ids)

    # must set seed as use dataloader twice
    set_seed(args.seed)

    tester = Tester(args)
    tester.testing()
