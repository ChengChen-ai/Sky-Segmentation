import torch
import torch.nn as nn

class Decom_Net(nn.Module):
    def __init__(self, name ,num_layers):
        super(Decom_Net, self).__init__()
        layers = []
        self.name = name
        layers.append(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=9, stride=1, padding=4))
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        img = torch.cat((torch.max(input=img, dim=1, keepdim=True)[0], img), dim=1)
        output = self.model(img)
        R, I = output[:,:3,:,:], output[:,3:4,:,:]
        return R, I

class Enhance_Net(nn.Module):
    def __init__(self, name):
        super(Enhance_Net, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU(True)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu3 = nn.ReLU(True)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up_relu2 = nn.ReLU(True)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.activation = nn.ReLU(True)

        self.fusion_conv = nn.Conv2d(in_channels=321, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Tanh()



    def forward(self, R, I):
        # enhance的输入是两个
        I = torch.cat((R,I),dim=1)
        h1 = self.conv1(I)
        h11 = self.relu1(h1)

        h2 = self.conv2(h11)
        h21 = self.relu2(h2)

        h3 = self.conv3(h21) ##[N,64,H/8,W/8]
        h31 = self.relu3(h3)

        h2_ = self.up_conv3(h31)
        h2_ = torch.cat((h2, h2_),dim=1) ##[N,128,H/4,W/4]
        h22 = self.up_relu3(h2_)

        h1_ = self.up_conv2(h22)
        h1_ = torch.cat((h1, h1_),dim=1) ##[N,128,H/2,W/2]
        h12 = self.up_relu2(h1_)

        x = self.up_conv1(h12)  ##[N,1,H,W]
        x = self.activation(x)

        c1 = nn.UpsamplingNearest2d(scale_factor=2)(h1_)
        c2 = nn.UpsamplingNearest2d(scale_factor=4)(h2_)
        c3 = nn.UpsamplingNearest2d(scale_factor=8)(h3)

        x = torch.cat([x,c1,c2,c3], dim=1)

        x = self.fusion_conv(x)
        x = self.final_conv(x)  ##[N,1,H,W]
        x = self.final_activation(x)

        return x
