# @Time    : 2022/4/6 14:54
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : metric.py
# @Software: PyCharm


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2022 Original Author PEIWEN PAN, original link is  https://github.com/GrokCV/SeRankDet/blob/master/utils/metric.py
# Copyright (c) 2024 Modified by Ruimin Huang
#
# Modifications:
# - Added f1_score for all threshold
# - Make it suitable for my project



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from tqdm import tqdm


class SigmoidMetric():
    def __init__(self, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # 其实是为了从numpy.ndarray的数据类型中转换过来
        mIoU = IoU.mean()
        # pixAcc：其实是PD，或者是召回率
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        # P map
        predict = (output > self.score_thresh).astype('int64')
        # TP sum: sum(P map == T map)
        pixel_correct = np.sum((predict == target) * (target > 0))
        
        # T sum
        pixel_labeled = np.sum(target > 0)
        
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass
        # P map
        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        # T map
        target = target.cpu().numpy().astype('int64')
        # TP map
        intersection = predict * (predict == target)
        
        # area_inter属于numpy.ndarray数据类型
        # count of TP map
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        # count of P map
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        # count of T map
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        # count of union map
        area_union = area_pred + area_lab - area_inter

        # 对布尔数组调用 .all() 会检查所有元素是否都为 True
        assert ( area_inter <= area_union ).all()

        return area_inter, area_union


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5, do_sigmoid=True):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()
        self.do_sigmoid = do_sigmoid

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels, self.nclass, self.score_thresh)
        # np.append([ [1], [2], [3] ],[ [4], [5], [6] ]) = [1 2 3 4 5 6], numpy.ndarray数据类型一样
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # 这是为了对多张图片取mean
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        # 回归到空的array
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        # P map
        if(self.do_sigmoid):
            predict = (F.sigmoid(output).cpu().detach().numpy() > score_thresh).astype('int64')  # P, 除了Swin_UNet之外使用这行
        else:
            predict = (output.cpu().detach().numpy() > score_thresh).astype('int64') # P, Swin_UNet使用这行，因为在Swin_UNet的后处理经过softmax函数，数值已经在0-1之间
        # T map
        target = target.cpu().detach().numpy().astype('int64')  # T
        # P=T map
        intersection = predict * (predict == target)  # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # area_inter属于numpy.ndarray数据类型
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter[0]

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred[0]

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab[0]

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union[0]

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins, do_sigmoid=True):  # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # 后来添加的
        self.reset()
        self.do_sigmoid = do_sigmoid

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = self.cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)
        # 但只计算了threshold为0.5的时候的f1_score
        f1_score = (2.0 * recall[int(self.bins/2)] * precision[int(self.bins/2)]) / (recall[int(self.bins/2)] + precision[int(self.bins/2)] + 0.00001)
        f1_score_all = (2.0 * recall * precision) / (recall + precision + 0.00001)

        return tp_rates, fp_rates, recall, precision, f1_score, f1_score_all

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    
    def cal_tp_pos_fp_neg(self, output, target, nclass, score_thresh):

        if(self.do_sigmoid):
            predict = (torch.sigmoid(output) > score_thresh).float() #除了Swin_UNet之外使用这行
        else:
            predict = ( output > score_thresh).float() # Swin_UNet使用这行，因为在Swin_UNet的后处理经过softmax函数，数值已经在0-1之间
        if len(target.shape) == 3:
            # 加一个维度 使得target与 output的size一致
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")
        
        # 要有整张图像的意识
        #  现在predict中高于阈值的部分为全1矩阵   target是GT
        intersection = predict * ((predict == target).float())
        # 对的预测为对的(OK)
        tp = intersection.sum()
        # 错的预测为对的 虚警像素数(OK)
        fp = (predict * ((predict != target).float())).sum()
        # 错的预测为错的(OK)，这里要考虑到1相等的地方，也要考虑到0相等的地方
        tn = ((1 - predict) * ((predict == target).float())).sum()
        # 对的预测为错的(OK)，这里也很好理解
        fn = (((predict != target).float()) * (1 - predict)).sum()
        # 标签中 阳性的个数
        pos = tp + fn
        # 标签中 阴性的个数
        neg = fp + tn
        # 检测出的个数
        class_pos = tp + fp
        return tp, pos, fp, neg, class_pos


# used in most projects
class SeRankDet_PD_FA():
    def __init__(self, nclass, bins):
        super(SeRankDet_PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.all_pixel = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        # self.cfg = cfg
        # 后来添加的
        self.reset()

    def update(self, preds, labels, size):

        # 先进行外围的threshold
        for iBin in range(self.bins + 1):
            # score_thresh = iBin * (255 / self.bins) # 用于0到255的范围
            score_thresh = (iBin + 0.0) / self.bins # 用于0到1的范围
            batch = preds.size()[0]
            # 再进行内围的batch，也就是一张image和一张label进行比较
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                # 不影响，因为已经在dataloader里面处理过了
                # predits = np.reshape(predits, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')  # P
                # 不影响，因为已经在dataloader里面处理过了
                # labelss = np.reshape(labelss, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))

                # 八连通区域聚类，聚类得到图像
                image = measure.label(predits, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_image = measure.regionprops(image)

                # 八连通区域聚类，聚类得到图像
                label = measure.label(labelss, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                # 叠加的每一对图片内部的指标，到下一张图片之前，要清空。
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []
                # 后面直接赋值，所以其实不用进行清空
                self.dismatch = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            self.image_area_match.append(area_image)

                            del coord_image[m]
                            break

                # FA是像素级别的比较
                # 面积在总图像面积里面，但不在已经匹配的面积里面，所以就是FA
                self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                self.FA[iBin] += np.sum(self.dismatch)
                self.all_pixel[iBin] += size[0] * size[1] # 单通道图像，不用考虑channel；并且，这里要严格区分iBin
                # PD是个数级别的比较，这里算的是个数，并没有直接使用distance_match
                self.PD[iBin] += len(self.distance_match)

    def get(self):

        # Final_FA = self.FA / ((self.cfg.data['crop_size'] * self.cfg.data['crop_size']) * img_num)
        # FA是像素级别的比较
        Final_FA = self.FA / self.all_pixel
        # PD是个数级别的比较
        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):
        # 持续累加变量1
        self.FA = np.zeros([self.bins + 1])
        # 持续累加变量2
        self.PD = np.zeros([self.bins + 1])
        
        # 后面自己添加的
        # 持续累加变量3 (直接算总数，可以不用考虑bin)，20250317改正，如果不考虑bin，那么all_pixel就会加bin次，所以要考虑bin。
        self.all_pixel = np.zeros(self.bins + 1)
        # 持续累加变量4 (其实也可以不用考虑bin，因为label的target数不受threshold影响)，20250317改正，需考虑，原因同上
        self.target = np.zeros(self.bins + 1)


# Not used
class BasicIRSTD_PD_FA():
    def __init__(self, nclass, bins):
        super(BasicIRSTD_PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.all_pixel = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        # self.cfg = cfg
        # 后来添加的
        self.reset()

    def update(self, preds, labels, size):

        # 先进行外围的threshold
        for iBin in range(self.bins + 1):
            # score_thresh = iBin * (255 / self.bins) # 不合理 not reasonable
            score_thresh = (iBin + 0.0) / self.bins # 更合理 more reasonable
            batch = preds.size()[0]
            # 再进行内围的batch，也就是一张image和一张label进行比较
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                # 不影响，因为已经在dataloader里面处理过了
                # predits = np.reshape(predits, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')  # P
                # 不影响，因为已经在dataloader里面处理过了
                # labelss = np.reshape(labelss, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))

                # 八连通区域聚类，聚类得到图像
                image = measure.label(predits, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_image = measure.regionprops(image)

                # 八连通区域聚类，聚类得到图像
                label = measure.label(labelss, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                # 叠加的每一对图片内部的指标，到下一张图片之前，要清空。
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []
                # 后面直接赋值，所以其实不用进行清空
                self.dismatch = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                true_img = np.zeros(predits.shape)
                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            # self.image_area_match.append(area_image)
                            true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1

                            del coord_image[m]
                            break

                # FA是像素级别的比较
                # 面积在总图像面积里面，但不在已经匹配的面积里面，所以就是FA
                # self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                # self.FA[iBin] += np.sum(self.dismatch)
                # !!! but not reasonable
                self.FA[iBin] += (predits - true_img).sum()
                self.all_pixel[iBin] += size[0] * size[1] # 单通道图像，不用考虑channel；并且，这里要严格区分iBin
                # PD是个数级别的比较，这里算的是个数，并没有直接使用distance_match
                self.PD[iBin] += len(self.distance_match)

    def get(self):

        # Final_FA = self.FA / ((self.cfg.data['crop_size'] * self.cfg.data['crop_size']) * img_num)
        # FA是像素级别的比较
        Final_FA = self.FA / self.all_pixel
        # PD是个数级别的比较
        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):
        # 持续累加变量1
        self.FA = np.zeros([self.bins + 1])
        # 持续累加变量2
        self.PD = np.zeros([self.bins + 1])
        
        # 后面自己添加的
        # 持续累加变量3 (直接算总数，可以不用考虑bin)，20250317改正，如果不考虑bin，那么all_pixel就会加bin次，所以要考虑bin。
        self.all_pixel = np.zeros(self.bins + 1)
        # 持续累加变量4 (其实也可以不用考虑bin，因为label的target数不受threshold影响)，20250317改正，需考虑，原因同上
        self.target = np.zeros(self.bins + 1)


# as a reference
class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.all_pixel = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        # self.cfg = cfg
        # 后来添加的
        self.reset()

    def update(self, preds, labels, size):

        # 先进行外围的threshold
        for iBin in range(self.bins + 1):
            # score_thresh = iBin * (255 / self.bins) # 不合理 not reasonable
            score_thresh = (iBin + 0.0) / self.bins # 更合理 more reasonable
            batch = preds.size()[0]
            # 再进行内围的batch，也就是一张image和一张label进行比较
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                # 不影响，因为已经在dataloader里面处理过了
                # predits = np.reshape(predits, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')  # P
                # 不影响，因为已经在dataloader里面处理过了
                # labelss = np.reshape(labelss, (self.cfg.data['crop_size'], self.cfg.data['crop_size']))

                # 八连通区域聚类，聚类得到图像
                image = measure.label(predits, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_image = measure.regionprops(image)

                # 八连通区域聚类，聚类得到图像
                label = measure.label(labelss, connectivity=2)
                # 将聚类得到的图像计算指标
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                # 叠加的每一对图片内部的指标，到下一张图片之前，要清空。
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []
                # 后面直接赋值，所以其实不用进行清空
                self.dismatch = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                # true_img = np.zeros(predits.shape)
                # more reasonable than BasicIRSTD_PD_FA
                true_img = np.zeros(labelss.shape)
                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            # self.image_area_match.append(area_image)

                            # true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                            # more reasonable than BasicIRSTD_PD_FA
                            true_img[coord_label[i].coords[:,0], coord_label[i].coords[:,1]] = 1

                            del coord_image[m]
                            break

                # FA是像素级别的比较
                # 面积在总图像面积里面，但不在已经匹配的面积里面，所以就是FA
                # self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                # self.FA[iBin] += np.sum(self.dismatch)
                # self.FA[iBin] += (predits - true_img).sum()
                
                # more reasonable than BasicIRSTD_PD_FA
                self.FA[iBin] += (predits * ((predits != true_img).astype('int64'))).sum()

                self.all_pixel[iBin] += size[0] * size[1] # 单通道图像，不用考虑channel；并且，这里要严格区分iBin
                # PD是个数级别的比较，这里算的是个数，并没有直接使用distance_match
                self.PD[iBin] += len(self.distance_match)

    def get(self):

        # Final_FA = self.FA / ((self.cfg.data['crop_size'] * self.cfg.data['crop_size']) * img_num)
        # FA是像素级别的比较，两个int相除，结果是float
        Final_FA = self.FA / self.all_pixel
        # PD是个数级别的比较
        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):
        # 持续累加变量1
        self.FA = np.zeros([self.bins + 1])
        # 持续累加变量2
        self.PD = np.zeros([self.bins + 1])
        
        # 后面自己添加的
        # 持续累加变量3 (直接算总数，可以不用考虑bin)，20250317改正，如果不考虑bin，那么all_pixel就会加bin次，所以要考虑bin。
        self.all_pixel = np.zeros(self.bins + 1)
        # 持续累加变量4 (其实也可以不用考虑bin，因为label的target数不受threshold影响)，20250317改正，需考虑，原因同上
        self.target = np.zeros(self.bins + 1)


if __name__ == '__main__':
    m1 = SigmoidMetric(score_thresh=0.5)
    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)

    m3 = ROCMetric(1,10)
    m4 = SeRankDet_PD_FA(1,10)
    
    for i in tqdm( range(100) ):
        
        pred = torch.rand(16, 1, 256, 256)
        target = torch.ones(16, 1, 256, 256)

        m1.update(pred, target)
        m2.update(pred, target)
        Pd_or_Recall, mIoU = m1.get()
        Single_IoU_List, nIoU = m2.get()
        print("Pd_or_Recall:{}\r\n mIoU:{}\r\n Single_IoU_List:{}\r\n nIoU:{}\r\n".format(Pd_or_Recall, mIoU, Single_IoU_List, nIoU))

        m3.update(pred, target)
        m4.update(pred, target, pred[0, 0].shape)
        tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = m3.get()
        print("tp_rates:{}\r\n fp_rates:{}\r\n recall:{}\r\n precision:{}\r\n f1_score:{}\r\n f1_score_all:{}\r\n".format(tp_rates, fp_rates, recall, precision, f1_score, f1_score_all))
        Final_FA, Final_PD = m4.get()
        print("Final_FA:{}\r\n Final_PD:{}\r\n".format(Final_FA, Final_FA))

# One of the result
'''
Pd_or_Recall:0.49990177154541016

mIoU:0.49990177154541016

Single_IoU_List:[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

nIoU:1.0

tp_rates:[1.         1.         1.         1.         1.         1.
 0.59449768 0.15265656 0.         0.         0.        ]

fp_rates:[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

recall:[1.         1.         1.         1.         1.         1.
 0.59449768 0.15265656 0.         0.         0.        ]

precision:[1.         1.         1.         1.         1.         1.
 1.         0.99999999 0.         0.         0.        ]

f1_score:0.9999949990713256

f1_score_all:[0.999995   0.999995   0.999995   0.999995   0.999995   0.999995
 0.7456818  0.26487548 0.         0.         0.        ]

Final_FA:[0.00000000e+00 0.00000000e+00 6.93581321e-07 1.22243708e-05
 1.04297291e-04 8.53278420e-04 3.61343731e-02 2.63937170e-02
 1.60249363e-02 6.25306910e-03 0.00000000e+00]

Final_PD:[0.00000000e+00 0.00000000e+00 6.93581321e-07 1.22243708e-05
 1.04297291e-04 8.53278420e-04 3.61343731e-02 2.63937170e-02
 1.60249363e-02 6.25306910e-03 0.00000000e+00]

'''