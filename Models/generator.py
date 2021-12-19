import torch.nn as nn
import ops
import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Dense_Block(nn.Module):
    def __init__(self, num_filter):
        super(Dense_Block, self).__init__()
        self.conv1 = ops.ConvBlock(num_filter, num_filter, stride=1)
        self.conv2 = ops.ConvBlock(num_filter, num_filter, stride=1)
        self.conv3 = ops.ConvBlock(2*num_filter, num_filter, stride=1)
    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        concat = torch.cat((conv1,conv2),1)
        output = self.conv3(concat)
        return output

class Generator(torch.nn.Module):
    def __init__(
        self, name, input_dim, num_filter, output_dim, num_resnet
    ):
        super(Generator, self).__init__()
        self.name = name
        self.pad = torch.nn.ReflectionPad2d(3)
        # Encoder
        self.conv11 = ops.ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0)

        self.dense_block21 = Dense_Block(num_filter)
        self.bn21 = torch.nn.InstanceNorm2d(num_filter)
        self.conv22 = ops.ConvBlock(num_filter, num_filter, kernel_size=1, padding=0, stride=1, activation='no_act',
                                    batch_norm=False)
        self.pool21 = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.conv23 = ops.ConvBlock(num_filter * 2, num_filter * 2, kernel_size=3, stride=1, padding=1)

        self.dense_block31= Dense_Block(num_filter * 2)
        self.bn31 = torch.nn.InstanceNorm2d(num_filter * 2)
        self.conv32 = ops.ConvBlock(num_filter * 2, num_filter * 2,  kernel_size=1, padding=0, stride=1,
                                    activation='no_act', batch_norm=False)
        self.pool31 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv33 = ops.ConvBlock(num_filter * 4, num_filter * 4, kernel_size=3, stride=1, padding=1)


        # Resnet blocks
        self.resnet_blocks = self._make_resnet_blocks(ops.ResnetBlock,num_filter,num_resnet)
        # Decoder
        self.deconv1 = ops.DeconvBlock(num_filter * 4, num_filter * 2, batch_norm=False)
        self.conv51 = ops.ConvBlock(num_filter * 4, num_filter * 2, kernel_size=1, stride=1, padding=0,
                                    batch_norm=False)

        self.deconv2 = ops.DeconvBlock(num_filter * 2, num_filter, batch_norm=False)
        self.conv61 = ops.ConvBlock(num_filter*2, num_filter, kernel_size=1, stride=1, padding=0,
                                    batch_norm=False)

        self.deconv3 = ops.ConvBlock(num_filter, output_dim, kernel_size=7, stride=1, padding=0,
                                     activation='tanh', batch_norm=False)

        ##concat
        self.concat2 = ops.ConvBlock(num_filter * 4, num_filter * 2, kernel_size=3, padding=1, stride=1, )
        self.concat3 = ops.ConvBlock(num_filter * 2, num_filter, kernel_size=3, padding=1, stride=1, )


    def _make_resnet_blocks(self, block, num_filters, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(num_filters * 4))
        return torch.nn.Sequential(*layers)

    def forward(self, fog):

        #Encoder
        enc1 = self.conv11(self.pad(fog))

        dense_block21 = self.dense_block21(enc1)
        bn21 = self.bn21(dense_block21)
        enc2 = self.conv22(bn21)
        enc2 = self.pool21(enc2)
        enc2 = torch.cat((self.pool21(dense_block21), enc2),1)
        enc2 = self.conv23(enc2)

        dense_block31 = self.dense_block31(enc2)
        bn31 = self.bn31(dense_block31)
        enc3 = self.conv32(bn31)
        enc3 = self.pool31(enc3)
        enc3 = torch.cat((self.pool31(dense_block31), enc3),1)
        enc3 = self.conv33(enc3)

        #Resnet blocks
        res = self.resnet_blocks(enc3)

        #Deconder
        dec1 = nn.UpsamplingBilinear2d(scale_factor=2)(res)
        concat2 = self.concat2(dec1)
        concat2 = torch.cat((concat2,enc2), 1)
        conv2 = self.conv51(concat2) + enc2

        dec2 = nn.UpsamplingBilinear2d(scale_factor=2)(conv2)

        concat3 = self.concat3(dec2)
        concat3 = torch.cat((concat3, enc1), 1)
        conv3 = self.conv61(concat3) + enc1

        fack_fogfree = self.deconv3(self.pad(conv3))

        return fack_fogfree
