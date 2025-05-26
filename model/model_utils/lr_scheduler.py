from torch.optim.lr_scheduler import _LRScheduler
import torch
import matplotlib.pyplot as plt

from warmup_scheduler import GradualWarmupScheduler

# Manual implementation
# 注意：以下函数，将影响将某层置零的冻结权重方式
class WarmDecayLR(_LRScheduler):
    def __init__(self, optimizer, baisc_lr, total_epochs, warm_up_epochs=2, min_lr=1e-6, last_epoch=-1, verbose="deprecated"):
        """
        自定义学习率调度器：支持学习率热身（Warm-Up）和衰减（Decay）。
        
        :self.basic_lr: 基础学习率，在此基础上乘以缩放因子
        :param optimizer: 被调度的优化器
        :param total_epochs: 总训练 epoch 数
        :param warm_up_epochs: 热身阶段的 epoch 数，默认为 0
        :param min_lr: 最小学习率，默认为 0
        :param last_epoch: 上一个 epoch，默认为 -1 表示从头开始
        """
        self.basic_lr = baisc_lr
        self.total_epochs = total_epochs
        self.warm_up_epochs = warm_up_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            # 热身阶段
            # decay_factor = self.last_epoch + 1 / self.warm_up_epochs
            decay_factor = self.last_epoch / self.warm_up_epochs
        else:
            # 衰减阶段
            adjusted_epoch = self.last_epoch - self.warm_up_epochs
            adjusted_total = self.total_epochs - self.warm_up_epochs + 1
            decay_factor = pow(1 - float(adjusted_epoch) / adjusted_total, 0.9)

        # return [ (decay_factor * (group["lr"] - self.min_lr) + self.min_lr ) for group in self.optimizer.param_groups]
        return [ (decay_factor * (self.basic_lr - self.min_lr) + self.min_lr ) for group in self.optimizer.param_groups]

# Manual implementation
# 注意：以下函数，将影响将某层置零的冻结权重方式
class WarmConstantDecayLR(_LRScheduler):
    def __init__(self, optimizer, baisc_lr, total_epochs, warm_up_epochs=2, constant_epochs=200, min_lr=1e-6, last_epoch=-1, verbose="deprecated"):
        """
        自定义学习率调度器：支持学习率热身（Warm-Up）和恒定（Constant_Epochs）以及衰减（Decay）。
        
        :self.basic_lr: 基础学习率，在此基础上乘以缩放因子
        :param optimizer: 被调度的优化器
        :param total_epochs: 总训练 epoch 数
        :param warm_up_epochs: 热身阶段的 epoch 数，默认为 0
        :constant_epochs: 恒定阶段的 epoch 数，默认为 0
        :param min_lr: 最小学习率，默认为 0
        :param last_epoch: 上一个 epoch，默认为 -1 表示从头开始
        """
        self.basic_lr = baisc_lr
        self.total_epochs = total_epochs
        self.warm_up_epochs = warm_up_epochs
        self.constant_epochs = constant_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            # 热身阶段
            # decay_factor = self.last_epoch + 1 / self.warm_up_epochs
            decay_factor = self.last_epoch / self.warm_up_epochs
        elif self.last_epoch < self.warm_up_epochs + self.constant_epochs:
            # 恒定阶段
            decay_factor = 1
        else:
            # 衰减阶段
            adjusted_epoch = self.last_epoch - self.warm_up_epochs - self.constant_epochs
            adjusted_total = self.total_epochs - self.warm_up_epochs - self.constant_epochs + 1
            decay_factor = pow(1 - float(adjusted_epoch) / adjusted_total, 0.9)

        # return [ (decay_factor * (group["lr"] - self.min_lr) + self.min_lr ) for group in self.optimizer.param_groups]
        return [ (decay_factor * (self.basic_lr - self.min_lr) + self.min_lr ) for group in self.optimizer.param_groups]



# Manual implementation
# 注意：以下函数，将影响将某层置零的冻结权重方式
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, num_epochs, warmup, power=0.9, warmup_epochs=5, last_epoch=-1, verbose="deprecated"):
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.warmup = warmup
        self.warmup_epoch = warmup_epochs if self.warmup else 0
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.warmup and self.last_epoch <= self.warmup_epoch:
            if self.last_epoch == 0:
                lr = self.base_lr / self.warmup_epoch
            else:
                lr = self.last_epoch * (self.base_lr / self.warmup_epoch)
        else:
            lr = self.base_lr * (1 - (self.last_epoch - self.warmup_epoch) / self.num_epochs) ** self.power       
        return [ lr for group in self.optimizer.param_groups]



# Manual implementation
# 注意：以下函数，将影响将某层置零的冻结权重方式
def lr_scheduler_WarmDecayLR(optimizer, baisc_lr, total_epochs, scheduler_settings, min_lr):
    scheduler = WarmDecayLR(optimizer, baisc_lr, total_epochs, scheduler_settings['warm_up_epochs'], min_lr)
    return scheduler


# Manual implementation
# 注意：以下函数，将影响将某层置零的冻结权重方式
def lr_scheduler_WarmConstantDecayLR(optimizer, baisc_lr, total_epochs, scheduler_settings, min_lr):
    scheduler = WarmConstantDecayLR(optimizer, baisc_lr, total_epochs, scheduler_settings['warm_up_epochs'], scheduler_settings['constant_epochs'], min_lr)
    return scheduler

# Manual implementation
def lr_scheduler_PolyLR(optimizer, base_lr, num_epochs, scheduler_settings):
    scheduler = PolyLR(optimizer, base_lr, num_epochs, scheduler_settings['warmup'], scheduler_settings['power'], scheduler_settings['warmup_epochs'])
    return scheduler

# Official implementation
def lr_scheduler_MultiStepLR(optimizer, scheduler_settings):
    scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['milestones'], gamma=scheduler_settings['gamma'])
    return scheduler


# Official implementation
def lr_scheduler_CosineAnnealingLR(optimizer, total_epochs, min_lr, scheduler_settings):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr, last_epoch=scheduler_settings['last_epoch'])
    return scheduler_cosine


# Official implementation
def lr_scheduler_CosineAnnealingLR_With_GradualWarmup(optimizer, total_epochs, min_lr, scheduler_settings):
    warmup_epochs = 10
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr, last_epoch=scheduler_settings['last_epoch'])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    return scheduler



if __name__ == '__main__':

    # epochs = 400
    # basic_lr = 0.05
    # min_lr = 1e-6
    # lrs = []

    # model = torch.nn.Linear(10, 1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=basic_lr)
    
    # scheduler_settings = {'warm_up_epochs': 2}
    # scheduler = lr_scheduler_WarmDecayLR(optimizer, basic_lr, epochs, scheduler_settings, min_lr)

    # scheduler_settings = {'warm_up_epochs': 2, 'constant_epochs': 210}
    # scheduler = lr_scheduler_WarmConstantDecayLR(optimizer, basic_lr, epochs, min_lr)

    # scheduler_settings = {'milestones': [180, 280], 'gamma': 0.5}
    # scheduler = lr_scheduler_MultiStepLR(optimizer, scheduler_settings)

    lrs = []
    basic_lr = 0.001
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=basic_lr)
    epochs = 1000
    min_lr = 1e-5
    scheduler_settings = {'last_epoch': -1}
    scheduler = lr_scheduler_CosineAnnealingLR_With_GradualWarmup(optimizer, epochs, min_lr, scheduler_settings)


    # lrs = []
    # basic_lr = 0.01
    # model = torch.nn.Linear(10, 1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=basic_lr)
    # epochs = 5000
    # min_lr = 0.01
    # scheduler_settings = {'last_epoch': -1}
    # scheduler = lr_scheduler_CosineAnnealingLR(optimizer, min(epochs,2000), min_lr, scheduler_settings)


    # lrs = []
    # basic_lr = 1e-2
    # model = torch.nn.Linear(10, 1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=basic_lr)
    # epochs = 5000
    # min_lr = 1e-6
    # scheduler_settings = {'last_epoch': -1}
    # scheduler = lr_scheduler_CosineAnnealingLR(optimizer, epochs, min_lr, scheduler_settings)


    # lrs = []
    # model = torch.nn.Linear(10, 1)
    # basic_lr = 0.0001
    # optimizer = torch.optim.Adam(model.parameters(), lr=basic_lr)
    # epochs = 1500
    # scheduler_settings = {'warmup': 'linear', 'power': 0.9, 'warmup_epochs': 5}
    # scheduler = lr_scheduler_PolyLR(optimizer=optimizer, base_lr=basic_lr, num_epochs=epochs, scheduler_settings=scheduler_settings)


    for epoch in range(epochs):
        optimizer.zero_grad()
        optimizer.step()

        # lr = scheduler.get_last_lr()[0]
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        
        scheduler.step()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), lrs, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler')
    plt.legend()
    plt.grid(True)
    plt.show()
