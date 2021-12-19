import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


def get_smooth(I, direction):
    # smooth
    weights = torch.tensor([[0., 0.],
                            [-1., 1.]]
                           ).cuda()
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 0, 1)
    if direction == 'x':
        weights = weights_x
    elif direction == 'y':
        weights = weights_y

    F = torch.nn.functional
    output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))  # stride, padding
    return output

def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    gradients_I_x = get_smooth(I ,'x')
    gradients_I_y = get_smooth(I ,'y')

    return torch.mean \
        (gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=(1,1), activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            conv = self.conv(x)
            out = self.bn(conv)
        else:
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding )
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        bn = torch.nn.InstanceNorm2d(num_filter)
        relu = torch.nn.ReLU(True)
        pad = torch.nn.ReflectionPad2d(1)

        self.resnet_block = torch.nn.Sequential(
            pad,
            conv1,
            bn,
            relu,
            pad,
            conv2,
            bn
        )

    def forward(self, x):
        out = self.resnet_block(x)
        return out
