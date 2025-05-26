import torch

def optimizer_set_adagrad(parameters, learning_rate):
    optimizer = torch.optim.Adagrad(params=parameters, lr=learning_rate, weight_decay=1e-4)
    return optimizer

def optimizer_set_adam(parameters, learning_rate):
    optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)
    return optimizer

def optimizer_set_adamw(parameters, learning_rate):
    optimizer = torch.optim.AdamW(params=parameters, lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))
    return optimizer

def optimizer_set_sgd(parameters, learning_rate):
    optimizer = torch.optim.SGD(params=parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    return optimizer
