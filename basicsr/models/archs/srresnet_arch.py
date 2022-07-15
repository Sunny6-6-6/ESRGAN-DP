from torch import nn as nn
from torch.nn import functional as F
import torch
from basicsr.models.archs import arch_util as arch_util


class upsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (2 ** 2), 3, stride=1, padding=1)  # r**2 x h x w -> r*h x r*w

        # 通道注意力
  #      self.fc1 = nn.Conv2d(int(in_channels * (2 ** 2)), int((in_channels * (2 ** 2)) / 16), (1, 1), stride=1)
  #      self.relu = nn.ReLU()
  #      self.fc2 = nn.Conv2d(int((in_channels * (2 ** 2)) / 16), int(in_channels * (2 ** 2)), (1, 1), stride=1)
        # 上采样
#        self.shuffler = nn.PixelShuffle(2)  # upscaling 2
 #       self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # 通道注意力
        # squeeze step
#        z = F.avg_pool2d(x, x.size()[2:])
        # excitation step
 #       s = torch.sigmoid(self.fc2(self.relu(self.fc1(z))))
  #      x = x + s.expand(x.size(0), x.size(1), x.size(2), x.size(3))

        return x


class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_block=16,
                 upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        # if self.upscale in [2, 3]:
        #     self.upconv1 = nn.Conv2d(num_feat,
        #                              num_feat * self.upscale * self.upscale, 3,
        #                              1, 1)
        #     self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        if self.upscale == 4:
            self.upconv1 = upsampleBlock(num_feat)
            self.upconv2 = upsampleBlock(num_feat)
         #   self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
         #   self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
      #  for i in range(int(self.upscale/2)):
      #      self.add_module('upsample'+str(i+1),upsampleBlock(num_feat))

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # # initialization
        arch_util.default_init_weights(
            [self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            arch_util.default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
       # elif self.upscale in [2, 3]:
        #    out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
       # for i in range(int(self.upscale/2)):
        #    out = self.__getattr__('upsample'+str(i+1))(out)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
 #       base = F.interpolate(
  #           x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
   #     out += base
        return out
