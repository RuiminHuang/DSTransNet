import torch
from torch import nn

from .ACM_UNet.model_ACM import ASKCResUNet

from .DSTransNet.DSTransNet import DSTransNet
from .DSTransNet.Config import get_DSTransNet_config

from .LW_IRSTNet.LW_IRST_ablation import LW_IRST_ablation

from .MLPNet.MLPNet import MLPNet

from .SCTransNet.SCTransNet import SCTransNet
from .SCTransNet.Config import get_SCTrans_config


from .model_utils.loss import SoftIoULoss, FocalLoss, DiceLoss, SLSIoULoss
from torch.nn.modules.loss import CrossEntropyLoss

class ACM_UNet(nn.Module):

    def __init__(self, mode):
        super(ACM_UNet, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.model = ASKCResUNet(layer_blocks=[4, 4, 4], channels=[8, 16, 32, 64], fuse_mode='AsymBi')
        elif self.mode == 'test':
            self.model = ASKCResUNet(layer_blocks=[4, 4, 4], channels=[8, 16, 32, 64], fuse_mode='AsymBi')
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = SoftIoULoss()
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
    

    
class DS_TransNet(nn.Module):

    def __init__(self, mode):
        super(DS_TransNet, self).__init__()

        self.config_vit = get_DSTransNet_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = DSTransNet(config=self.config_vit, input_channels=3, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = DSTransNet(config=self.config_vit, input_channels=3, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this, because it returen tuple
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss



class LW_IRSTNet(nn.Module):

    def __init__(self, mode):
        super(LW_IRSTNet, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
        elif self.mode == 'test':
            self.model = LW_IRST_ablation(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3))
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = SoftIoULoss()
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss



class MLP_Net(nn.Module):

    def __init__(self, mode, img_size):
        super(MLP_Net, self).__init__()
        self.mode = mode
        self.img_size = img_size
        if self.mode == 'train':
            self.model = MLPNet(img_size=self.img_size, patch_size=4, in_chans=3, num_classes=1,
                   embed_dim=64, layers=[2, 2, 2, 2], drop_rate=0.5,
                  norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True)
        elif self.mode == 'test':
            self.model = MLPNet(img_size=self.img_size, patch_size=4, in_chans=3, num_classes=1,
                   embed_dim=64, layers=[2, 2, 2, 2], drop_rate=0.5,
                  norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.num_classes = 2

        self.bce_loss = nn.BCELoss(reduction='mean')
        self.dice_loss = DiceLoss(n_classes=self.num_classes)
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, pred, gt_mask):
        
        # singlel channel for BCE
        pred_single_BCE =  pred
        # double channel for Dice
        target_channel = pred
        background_channel = 1 - target_channel # 相当于已经做了softmax
        pred_double_Dice = torch.cat([background_channel, target_channel], dim=1)

        # construct mask
        gt_mask_BCE = gt_mask
        gt_mask_Dice = gt_mask.squeeze(1)
        
        # calculate loss
        loss_bce = self.bce_loss(pred_single_BCE, gt_mask_BCE) # Attention：不使用CrossEntropyLoss函数是为了避免再次使用softmax
        loss_dice = self.dice_loss(pred_double_Dice, gt_mask_Dice, softmax=False)  # Atention：已经加起来为1，不需要再softmax
        loss = 0.5 * loss_bce + 0.5 * loss_dice
        return loss



class SC_TransNet(nn.Module):

    def __init__(self, mode):
        super(SC_TransNet, self).__init__()

        self.config_vit = get_SCTrans_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = SCTransNet(config=self.config_vit, n_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = SCTransNet(config=self.config_vit, n_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        # only use this
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


# Using python -m DSTransNet.model.model to test the model
if __name__ == '__main__':

    model = DS_TransNet(mode='train')

    input_tensor = torch.randn((1, 3, 512, 512))

    outputs = model(input_tensor)
    
    # print(f'Output shape: {outputs.shape}')
    for output in outputs:
        print(type(output))
        print(output.shape)
