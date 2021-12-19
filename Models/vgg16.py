import torch
from torchvision import models
import torch.nn as nn
from ops import MeanShift


class Vgg16(nn.Module):
    def __init__(self,name, requires_grad=True, rgb_range=1):
        super(Vgg16,self).__init__()
        self.name = name
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.name = name
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()

        for x in range(10):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 37):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, input):
        x = self.sub_mean(input)
        x = self.slice1(x)
        x_lv1 = x
        # x_lv1 = self.conv1(x_lv1)
        x = self.slice2(x)
        x_lv2 = x
        return x_lv1,x_lv2




if __name__ == '__main__':
    pass
