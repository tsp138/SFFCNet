
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNReLU(nn.Module):
    def __init__(self, C_out, affine=True):
        super(BNReLU, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False))


    def forward(self, x):
        return self.op(x)

class GESD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GESD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(out_channels*3, out_channels, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)
        self.bnReLu1 = BNReLU(out_channels)
        self.bnReLu2 = BNReLU(out_channels)

        # 创建一个可训练的卷积核参数，形状为 (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        # 初始化第一行的权重为可训练参数
        nn.init.xavier_uniform_(self.weight[:, :, :, :])

    def forward(self, x):
        b,c,h,w=self.weight.size()
        weight = self.weight.clone()

        Sob0=weight[:, :, 0, 0]
        Sob90=weight[:, :, 0, 2]

        GL=weight[:, :, 2, 2]

        weight_0= torch.zeros(b, c, h, w).cuda()

        weight_0[:, :,0, 0] = -Sob0
        weight_0[:, :,0, 1] = -Sob0*2
        weight_0[:, :,0, 2] = -Sob0
        weight_0[:, :,1, 0] = 0
        weight_0[:, :,1, 1] = 0
        weight_0[:, :,1, 2] = 0
        weight_0[:, :,2, 0] = Sob0
        weight_0[:, :,2, 1] = Sob0*2
        weight_0[:, :,2, 2] = Sob0
        x_0=F.conv2d(x, weight_0, stride=self.stride, padding=self.padding)

        weight_90 = torch.zeros(b, c, h, w).cuda()
        weight_90[:, :, 0, 0] = Sob90
        weight_90[:, :, 0, 1] = 0
        weight_90[:, :, 0, 2] = -Sob90
        weight_90[:, :, 1, 0] = Sob90*2
        weight_90[:, :, 1, 1] = 0
        weight_90[:, :, 1, 2] = -Sob90*2
        weight_90[:, :, 2, 0] = Sob90
        weight_90[:, :, 2, 1] = 0
        weight_90[:, :, 2, 2] = -Sob90
        x_90=F.conv2d(x, weight_90, stride=self.stride, padding=self.padding)

        a1 =  GL * 4
        a2 = -a1 / 8
        a3 = -a1 / 16

        # kernel_Gaussian_Laplacian=torch.zeros(b,c,5,5)
        kernel_Gaussian_Laplacian=torch.zeros(b,c,5,5).cuda()
        kernel_Gaussian_Laplacian[:, :,0, 2] = a3
        kernel_Gaussian_Laplacian[:, :,1, 1] = a3
        kernel_Gaussian_Laplacian[:, :,1, 2] = a2
        kernel_Gaussian_Laplacian[:, :,1, 3] = a3
        kernel_Gaussian_Laplacian[:, :,2, 0] = a3
        kernel_Gaussian_Laplacian[:, :,2, 1] = a2
        kernel_Gaussian_Laplacian[:, :,2, 2] = a1
        kernel_Gaussian_Laplacian[:, :,2, 3] = a2
        kernel_Gaussian_Laplacian[:, :,2, 4] = a3
        kernel_Gaussian_Laplacian[:, :,3, 1] = a3
        kernel_Gaussian_Laplacian[:, :,3, 2] = a2
        kernel_Gaussian_Laplacian[:, :,3, 3] = a3
        kernel_Gaussian_Laplacian[:, :,4, 2] = a3

        x_GL=F.conv2d(x, kernel_Gaussian_Laplacian, stride=self.stride, padding=2)

        x_out = torch.cat((x_0,x_90,x_GL), dim=1)
        x_high = self.conv1(x_out)
        x_low = (1-self.sigmoid(x_high))*self.conv2(x)

        return self.bnReLu1(x_high), self.bnReLu2(x_low)