import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from networks.MobileNetV2 import MobileNetV2
from torch.hub import load_state_dict_from_url
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from thop import profile
from networks.separable_convolutions import DepthwiseSeparableConv

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:

            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([

            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, dilation=dilation),

            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=None, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s, d
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]


        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s, d in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                dilation = d if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, dilation=d))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            if idx in [1, 3, 6, 13, 17]:
                res.append(x)
        return res

    def mobilenet_v2(pretrained=True, progress=True, **kwargs):
        model = MobileNetV2(**kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            print("loading imagenet pretrained mobilenetv2")
            model.load_state_dict(state_dict, strict=False)
            print("loaded imagenet pretrained mobilenetv2")
        return model


def init_weights(m):

    if isinstance(m, nn.Conv2d):

        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SobelModule(nn.Module):
    def __init__(self, input_channel, mode='high_boost_filtering', parameter_a=1, parameter_k=0.5,
                 ):

        super(SobelModule, self).__init__()
        self.mode = mode
        self.channel = input_channel
        self.A = parameter_a
        self.K = parameter_k

        # Gaussian Smooth
        kernel_gaussian_smooth = [[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]
        kernel_smooth = torch.FloatTensor(kernel_gaussian_smooth).expand(self.channel, self.channel, 3, 3)
        self.weight_smooth = nn.Parameter(data=kernel_smooth, requires_grad=False)

        # Isotropic Sobel
        kernel_isotropic_sobel_direction_1 = [[1, math.sqrt(2), 1],
                                              [0, 0, 0],
                                              [-1, -math.sqrt(2), -1]]
        kernel_isotropic_sobel_direction_2 = [[0, 1, math.sqrt(2)],
                                              [-1, 0, 1],
                                              [-math.sqrt(2), -1, 0]]
        kernel_isotropic_sobel_direction_3 = [[-1, 0, 1],
                                              [-math.sqrt(2), 0, math.sqrt(2)],
                                              [-1, 0, 1]]
        kernel_isotropic_sobel_direction_4 = [[math.sqrt(2), 1, 0],
                                              [1, 0, -1],
                                              [0, -1, -math.sqrt(2)]]

        kernel_1 = torch.FloatTensor(kernel_isotropic_sobel_direction_1).expand(self.channel, self.channel, 3, 3)
        kernel_2 = torch.FloatTensor(kernel_isotropic_sobel_direction_2).expand(self.channel, self.channel, 3, 3)
        kernel_3 = torch.FloatTensor(kernel_isotropic_sobel_direction_3).expand(self.channel, self.channel, 3, 3)
        kernel_4 = torch.FloatTensor(kernel_isotropic_sobel_direction_4).expand(self.channel, self.channel, 3, 3)
        kernel_5 = -1 * kernel_1
        kernel_6 = -1 * kernel_2
        kernel_7 = -1 * kernel_3
        kernel_8 = -1 * kernel_4
        self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel_5, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel_6, requires_grad=False)
        self.weight_7 = nn.Parameter(data=kernel_7, requires_grad=False)
        self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)

    def forward(self, x):
        origin = x
        x_result = x
        x1 = F.conv2d(x, self.weight_1, stride=1, padding=1)
        x2 = F.conv2d(x, self.weight_2, stride=1, padding=1)
        x3 = F.conv2d(x, self.weight_3, stride=1, padding=1)
        x4 = F.conv2d(x, self.weight_4, stride=1, padding=1)
        x5 = F.conv2d(x, self.weight_5, stride=1, padding=1)
        x6 = F.conv2d(x, self.weight_6, stride=1, padding=1)
        x7 = F.conv2d(x, self.weight_7, stride=1, padding=1)
        x8 = F.conv2d(x, self.weight_8, stride=1, padding=1)
        x_high_frequency = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) / 8

        if self.mode == 'filtering':
            x_result = x_high_frequency
        elif self.mode == 'high_boost_filtering':
            x_result = self.A * x + self.K * x_high_frequency
        return origin, x_result


class EdgeEnhance(nn.Module):

    def __init__(self, inplanes, outplanes, ratio=16):
        super(EdgeEnhance, self).__init__()


        self.edge = SobelModule(inplanes)
        self.con_1x1 = nn.Conv2d(inplanes, outplanes, 1, bias=False)


        self.sconv1 = nn.Sequential(
            DepthwiseSeparableConv(inplanes, inplanes),

        )
        # -------------------------------------------------
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Conv2d(inplanes * 3, outplanes, kernel_size=1, bias=False)

        self.offset_conv = nn.Conv2d(outplanes, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(outplanes, outplanes, kernel_size=3, padding=1)

        self.ca = CoordinateAttention(outplanes, outplanes)

        #-------------------------------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        origin, edge = self.edge(x)
        edge=self.con_1x1(edge)
        x3=self.conv3x3(x)
        x5=self.conv5x5(x)
        x7=self.conv7x7(x)
        fused = torch.cat([x3, x5, x7], dim=1)
        fused = self.fusion(fused)
        offset = self.offset_conv(fused)
        deform_feat = self.deform_conv(fused, offset)
        out = self.ca(deform_feat)
        out = out+edge

        out = self.conv(out)

        return out

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out
class DE(nn.Module):


    def __init__(self, channel_dim):
        super(DE, self).__init__()

        self.channel_dim = channel_dim

        self.sconv2 = nn.Conv2d(self.channel_dim * 2, self.channel_dim * 2, kernel_size=3, padding=1)
        self.Conv1 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=self.channel_dim, kernel_size=1, stride=1,
                         padding=0)
        self.Conv2 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=self.channel_dim, kernel_size=1, stride=1,
                               padding=0)
        self.BN1 = nn.BatchNorm2d(self.channel_dim * 2)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channel_dim)
        self.sa = SpatialAttention()

    def forward(self, x1, x2):
        f_d = torch.abs(x1 - x2)
        f_c = torch.cat((x1, x2), dim=1)
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.sconv2(f_c))))))
        c_out = self.ca(z_c) * z_c
        s_out = self.sa(f_d) * f_d
        out = c_out * s_out
        return out


def cosine_similarity_map(feat1, feat2, eps=1e-6, apply_sigmoid=True):
    feat1_norm = F.normalize(feat1, p=2, dim=1, eps=eps)
    feat2_norm = F.normalize(feat2, p=2, dim=1, eps=eps)
    sim = (feat1_norm * feat2_norm).sum(dim=1, keepdim=True)
    if apply_sigmoid:
        sim = torch.sigmoid(sim)
    return sim

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UFF_Fusion(nn.Module):
    def __init__(self, in_channels,out_channels, reduction=16):
        super(UFF_Fusion, self).__init__()

        self.se = SELayer(out_channels, reduction=8)
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    def forward(self, feat_low, feat_high):

        feat_low=self.upsample(feat_low)
        sim_map = cosine_similarity_map(feat_low, feat_high)
        feat_weighted = (feat_low + feat_high)* sim_map
        sum= feat_low+feat_high+feat_weighted
        out = self.se(sum)

        return out

class NewNetwork(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super(NewNetwork, self).__init__()

        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)

        self.upsample = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.eh1 = EdgeEnhance(16, 16)
        self.eh2 = EdgeEnhance(24, 24)
        self.eh3 = EdgeEnhance(32, 32)
        self.eh4 = EdgeEnhance(96, 96)
        self.eh5 = EdgeEnhance(320, 320)


        self.Up_eh1 = EdgeEnhance(320, 96)
        self.uff1 = UFF_Fusion(96, 96)
        self.Up_eh2 = EdgeEnhance(96, 32)
        self.uff2 = UFF_Fusion(96, 32)
        self.Up_eh3 = EdgeEnhance(32, 24)
        self.uff3 = UFF_Fusion(24, 24)
        self.Up_eh4 = EdgeEnhance(24, 16)
        self.uff4 = UFF_Fusion(16, 16)
        self.Conv_1x1 = nn.Conv2d(16, out_ch, kernel_size=1, stride=1, padding=0)



        self.de1 = DE(16)
        self.de2 = DE(24)
        self.de3 = DE(32)
        self.de4 = DE(96)
        self.de5 = DE(320)




    def forward(self, x):

        x1 = x[:, :3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        x1_1 = self.eh1(x1_1)
        x2_1 = self.eh1(x2_1)
        x1 = self.de1(x1_1, x2_1)
        x1_2 = self.eh2(x1_2)
        x2_2 = self.eh2(x2_2)
        x2 = self.de2(x1_2, x2_2)
        x1_3 = self.eh3(x1_3)
        x2_3 = self.eh3(x2_3)
        x3 = self.de3(x1_3, x2_3)

        x1_4 = self.eh4(x1_4)
        x2_4 = self.eh4(x2_4)
        x4 = self.de4(x1_4, x2_4)
        x1_5 = self.eh5(x1_5)
        x2_5 = self.eh5(x2_5)
        x5 = self.de5(x1_5, x2_5)


        d5 = self.Up_eh1(x5)
        u1=self.uff1(d5,x4)
        d4 = self.Up_eh2(u1)
        u2=self.uff2(d4,x3)
        d3 = self.Up_eh3(u2)
        u3=self.uff3(d3,x2)
        d2=self.Up_eh4(u3)
        u4=self.uff4(d2,x1)
        d1 = self.upsample(u4)
        d1 = self.Conv_1x1(d1)
        pred = nn.Sigmoid()(d1)

        return pred, pred


    def init_weights(self):
        self.Conv1_1.apply(init_weights)
        self.eh1.apply(init_weights)
        self.eh2.apply(init_weights)
        self.eh3.apply(init_weights)
        self.eh4.apply(init_weights)
        self.eh5.apply(init_weights)
        self.Up_eh1.apply(init_weights)
        self.Up_eh2.apply(init_weights)
        self.Up_eh3.apply(init_weights)
        self.Up_eh4.apply(init_weights)
        self.de2.apply(init_weights)
        self.de3.apply(init_weights)
        self.de4.apply(init_weights)
        self.de5.apply(init_weights)


