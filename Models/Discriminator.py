import torch
import ops
'''
 Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
'''


class Discriminator(torch.nn.Module):
    def __init__(self, name, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()
        self.name=name
        conv1 = ops.ConvBlock(input_dim, num_filter // 4, kernel_size=3, stride=1, padding=1, activation='lrelu')
        conv2 = ops.ConvBlock(num_filter // 4, num_filter // 2,kernel_size=3, stride=2, padding=1, activation='lrelu')
        conv3 = ops.ConvBlock(num_filter // 2, num_filter , kernel_size=3, stride=2, padding=1, activation='lrelu')
        conv4 = ops.ConvBlock(num_filter , num_filter * 2, kernel_size=3, stride=2, padding=1, activation='lrelu')
        conv5 = ops.ConvBlock(num_filter * 2, num_filter * 2, kernel_size=3, stride=2, padding=1, activation='lrelu',)
        conv6 = ops.ConvBlock(num_filter * 2, num_filter * 2, kernel_size=3, stride=2, padding=1, activation='lrelu', )
        self.liear = torch.nn.Linear(num_filter * 2 * 16**2, output_dim)

        self.conv_blocks = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0],-1)

        out = self.liear(out)
        return out ###[b,1,h/16,w/16]
