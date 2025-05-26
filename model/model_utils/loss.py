import torch
import torch.nn as nn

# SoftIoULoss 的 PyTorch 实现
class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        smooth = 1
        intersection = pred * target
        
        # 这里保留了BatchSize维度
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        # 对所有的Batch_Size取平均
        loss = 1 - torch.mean(loss)
        return loss


# Focal Loss 的 PyTorch 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        # 计算 logits 的 sigmoid 概率
        p = torch.sigmoid(inputs)

        # 计算交叉熵损失，直接处理input，所以必须用with_logits
        # 返回的还是[BCHW]
        # 注意，这里inputs和targets的shape必须一致，并且为BCHW，C为1表示对1个类别进行二分类，binary_cross_entropy函数同理
        # 这一点和cross_entropy有明显区别，因为cross_entropy没要求inputs和targets的shape必须一致
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算 Focal Loss
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha * ((1 - p) ** self.gamma) * bce_loss

        # 根据 reduction 参数进行损失归约
        # 直接对一整个[BCHW]的数据进行reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss




# 注意，这里的target为0~n_classes-1的整数组成的单通道的Mask，在我们的数据预处理中，0~255被处理成为了0~1，凑巧0代表背景，1(255)代表目标，所以可以直接用。
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes



class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def LLoss(self, pred, target):
            
            loss = torch.tensor(0.0, requires_grad=True).to(pred)

            batch_size = pred.shape[0]
            h = pred.shape[2]
            w = pred.shape[3]        
            x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
            y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
            smooth = 1e-8
            
            for i in range(batch_size):  

                pred_centerx = (x_index*pred[i]).mean()
                pred_centery = (y_index*pred[i]).mean()

                target_centerx = (x_index*target[i]).mean()
                target_centery = (y_index*target[i]).mean()
            
                angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                                - torch.arctan((target_centery) / (target_centerx + smooth))))

                pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
                target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
                
                length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
            
                loss = loss + (1 - length_loss + angle_loss) / batch_size
            
            return loss

    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        lloss = self.LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss



# 注意，这里的target为0~n_classes-1的整数组成的单通道的Mask，在我们的数据预处理中，0~255被处理成为了0~1，凑巧0代表背景，1(255)代表目标，所以可以直接用。此时mask的shape为[B, H, W]
class CrossEntropyLoss_Warp(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_Warp, self).__init__()

    def forward(self, inputs, target):
        return nn.functional.cross_entropy(inputs, target)



if __name__ == '__main__':

    inputs = torch.rand(16, 1, 256, 256, requires_grad=True)
    targets = torch.ones(16, 1, 256, 256, dtype=torch.float32)

    print("input:")
    print(inputs.shape)
    print("target:")
    print(targets.shape)


    criterion = SoftIoULoss()
    loss = criterion(inputs, targets)
    print("SoftIoULoss:")
    print(loss.shape)
    print(loss)


    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    loss = criterion(inputs, targets)
    print("FocalLoss:")
    print(loss.shape)
    print(loss)


    n_classes = 2
    batch_size = 16
    criterion_dice = DiceLoss(n_classes)
    criterion_ce = CrossEntropyLoss_Warp()
    inputs = torch.rand(batch_size, n_classes, 256, 256, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, 256, 256)).float()
    print(inputs.shape)
    print(targets.shape)
    loss = criterion_dice(inputs, targets)
    print(f'Dice Loss: {loss}')
    loss = criterion_ce(inputs, targets[:].long())# same with: loss = criterion_ce(inputs, targets.long())
    print(f'CE Loss: {loss}')
