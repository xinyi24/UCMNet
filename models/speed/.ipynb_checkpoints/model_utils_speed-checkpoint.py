# ------------------------------------------------------------------------------
# Written by Yiwen Bai (wen1109@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False



class BasicBlock_MC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False, dropout_p=0.1):
        super(BasicBlock_MC, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu
        
        # Dropout rate for MC Dropout
        self.dropout_p = dropout_p

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply dropout after the first convolution
        out = F.dropout(out, p=self.dropout_p, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply dropout after the second convolution
        out = F.dropout(out, p=self.dropout_p, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        # print(x.shape)#torch.Size([4, 512, 16, 24])
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        # print(out.shape)#torch.Size([4, 128, 16, 24])
        return out


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SCE(nn.Module):
    def __init__(self, in_channels = 512, out_channels= 128 , grids=(6, 3, 2, 1)):
        super(SCE, self).__init__()

        self.reduce_channel = ConvBNReLU(in_channels, out_channels,1,1,0)
        self.grids = grids
        print('grid ar near',self.grids)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_1', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_2', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_3', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_4', ConvBNReLU(out_channels, out_channels,1,1,0))

        self.upsampling_method = lambda x, size: F.interpolate(x, size, mode='nearest')

        self.spatial_attention = nn.Sequential(
            ConvBNReLU(out_channels * 4, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, 4, kernel_size=1, bias=False), ##
            nn.Sigmoid()
        )
        self.keras_init_weight()
        self.spatial_attention[1].weight.data.zero_()


    def forward(self, x):

        size = x.size()[2:]

        ar = size[1] / size[0]
        x = self.reduce_channel(x) # 降维

        context = []
        for i in range(len(self.grids)):
            grid_size = (self.grids[i], max(1, round(ar * self.grids[i])))
            # grid_size = (self.grids[i], self.grids[i])
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            x_pooled = self.spp[i].forward(x_pooled)
            x_pooled = self.upsampling_method(x_pooled,size)
            context.append(x_pooled)
            # out = out + x_pooled

        spatial_att = self.spatial_attention(torch.cat(context,dim=1))  + 1 ## truple 4

        x = x + context[0] * spatial_att[:, 0:1, :, :] + context[1] * spatial_att[:, 1:2, :, :]  \
            + context[2] * spatial_att[:, 2:3, :, :] + context[3] * spatial_att[:, 3:4, :, :]


        return x
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 128  # 修改为128
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 修改输出通道数为128
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DASPPConv, self).__init__(*modules)


class DASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.kernel_size = kernel_size

    def forward(self, x):
        size = x.shape[-2:]
        x = super(DASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(DASPP, self).__init__()
        out_channels = 128  # 修改为128
        modules = []
        
        # 原始的1x1卷积和3x3卷积组合
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        
        # 三个不同的池化大小和对应的atrous rate
        modules.append(DASPPConv(in_channels, out_channels, rate1))
        modules.append(DASPPConv(in_channels, out_channels, rate2))
        modules.append(DASPPConv(in_channels, out_channels, rate3))
        
        # 三个不同的池化大小
        modules.append(DASPPPooling(in_channels, out_channels, kernel_size=3))
        modules.append(DASPPPooling(in_channels, out_channels, kernel_size=5))
        modules.append(DASPPPooling(in_channels, out_channels, kernel_size=7))

        self.convs = nn.ModuleList(modules)

        # 修改输出通道数为128
        self.project = nn.Sequential(
            nn.Conv2d(6 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class PPContextModule(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes, align_corners=False):
        super(PPContextModule, self).__init__()

        # Stage list, each stage is a sequence of adaptive pooling and convolution layers
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        # Final convolution layer after all stages to output the desired number of channels (128)
        self.conv_out = nn.Conv2d(in_channels=inter_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1)
        
        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        # Create each stage consisting of adaptive average pooling and 1x1 convolution
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]  # Get the height and width of the input

        # Iterate through each stage
        for stage in self.stages:
            x = stage(input)
            # Upsample to the original input size using bilinear interpolation
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        # Apply the final convolution layer
        out = self.conv_out(out)
        return out


class DConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(DConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,  # 使用空洞卷积
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SCM(nn.Module):
    def __init__(self, in_channels=512, out_channels=128, grids=(6, 3, 2, 1), dilation=2):
        super(SCM, self).__init__()

        self.reduce_channel = DConvBNReLU(in_channels, out_channels, 1, 1, 0)
        self.grids = grids
        print('Grid sizes:', self.grids)

        self.spp = nn.Sequential()
        for i in range(4):
            self.spp.add_module(f'spp_{i + 1}', DConvBNReLU(out_channels, out_channels, 1, 1, 0))

        self.upsampling_method = lambda x, size: F.interpolate(x, size, mode='nearest')

        self.spatial_attention = nn.Sequential(
            DConvBNReLU(out_channels * 4, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, 4, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空洞卷积层：为每个分支添加不同的空洞卷积
        self.dilated_conv_0 = DConvBNReLU(out_channels, out_channels, 3, 1, 2, dilation=dilation)  # 3x3 空洞卷积
        self.dilated_conv_1 = DConvBNReLU(out_channels, out_channels, 3, 1, 2, dilation=dilation)
        self.dilated_conv_2 = DConvBNReLU(out_channels, out_channels, 3, 1, 2, dilation=dilation)
        self.dilated_conv_3 = DConvBNReLU(out_channels, out_channels, 3, 1, 2, dilation=dilation)

        self.keras_init_weight()
        self.spatial_attention[1].weight.data.zero_()

    def forward(self, x):
        size = x.size()[2:]

        ar = size[1] / size[0]
        x = self.reduce_channel(x)

        context = []
        for i in range(len(self.grids)):
            grid_size = (self.grids[i], max(1, round(ar * self.grids[i])))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            x_pooled = self.spp[i].forward(x_pooled)
            x_pooled = self.upsampling_method(x_pooled, size)
            context.append(x_pooled)

        spatial_att = self.spatial_attention(torch.cat(context, dim=1)) + 1

        # 在空间注意力之后，分别进行空洞卷积
        context_0 = context[0] * spatial_att[:, 0:1, :, :]
        context_1 = context[1] * spatial_att[:, 1:2, :, :]
        context_2 = context[2] * spatial_att[:, 2:3, :, :]
        context_3 = context[3] * spatial_att[:, 3:4, :, :]

        # 分别应用空洞卷积
        context_0 = self.dilated_conv_0(context_0)
        context_1 = self.dilated_conv_1(context_1)
        context_2 = self.dilated_conv_2(context_2)
        context_3 = self.dilated_conv_3(context_3)

        # 最终输出
        x = x + context_0 + context_1 + context_2 + context_3

        return x

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        # print(x.shape)#torch.Size([4, 64, 128, 192])
        # print(y.shape)#torch.Size([4, 64, 64, 96])
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        # print(x.shape)#torch.Size([4, 64, 128, 192])
        return x


class ConvBNReLU3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBNReLU3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BFC(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BFC, self).__init__()

        # 保持输入和输出通道数一致
        self.conv_8 = ConvBNReLU3(32, 32, 3, 1, 1)  # 输入输出通道保持一致
        self.cp1x1 = nn.Conv2d(32, 32, 1, bias=False)  # 使用32通道进行1x1卷积

        self.conv_32 = ConvBNReLU3(32, 32, 3, 1, 1)  # 输入输出通道保持一致

        self.sp1x1 = nn.Conv2d(32, 32, 1, bias=False)  # 使用32通道进行1x1卷积

        # 设置 groups 参数
        self.groups = 2
        print('groups', self.groups)

        # 更新conv_offset的通道数，以适应输入和输出的通道数
        self.conv_offset = nn.Sequential(
            ConvBNReLU3(64, 64, 1, 1, 0),  # 输入通道为64, 输出通道保持一致
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.keras_init_weight()

        # 初始化卷积层的权重
        self.conv_offset[1].weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, cp, sp):
        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征，输入输出通道一致，尺寸 1/8 256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)

        # x_8
        cp = self.conv_8(cp)

        # 将 cp1x1 和 sp1x1 与 conv_offset 合并，以减少计算量
        cp1x1 = self.cp1x1(cp)
        sp1x1 = self.sp1x1(sp)

        conv_results = self.conv_offset(torch.cat([cp1x1, sp1x1], 1))

        # 重塑sp和cp以适应offset操作
        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0:self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)

        # 创建网格用于grid_sample
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        # 通过网格偏移来调整sp和cp
        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        # 使用grid_sample进行采样
        cp = F.grid_sample(cp, grid_l, align_corners=True)
        sp = F.grid_sample(sp, grid_h, align_corners=True)

        # 恢复原始尺寸
        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        # 计算注意力机制
        att = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        
        # att = torch.sigmoid(conv_results[:, self.groups * 4:, :, :])  # 使用 Sigmoid 激活函数

        # 最终合并特征
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


class BSC(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BSC, self).__init__()

        # 修改输入和输出通道数为 64
        self.conv_8 = ConvBNReLU3(64, 64, 3, 1, 1)  # 输入输出通道保持一致
        self.cp1x1 = nn.Conv2d(64, 64, 1, bias=False)  # 使用64通道进行1x1卷积

        self.conv_32 = ConvBNReLU3(64, 64, 3, 1, 1)  # 输入输出通道保持一致

        self.sp1x1 = nn.Conv2d(64, 64, 1, bias=False)  # 使用64通道进行1x1卷积

        # 设置 groups 参数
        self.groups = 2
        print('groups', self.groups)

        # 更新conv_offset的通道数，以适应输入和输出的通道数
        self.conv_offset = nn.Sequential(
            ConvBNReLU3(128, 128, 1, 1, 0),  # 输入通道为128, 输出通道保持一致
            nn.Conv2d(128, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)
        )

        self.keras_init_weight()

        # 初始化卷积层的权重
        self.conv_offset[1].weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, cp, sp):
        n, _, out_h, out_w = cp.size()

        # x_32
        sp = self.conv_32(sp)  # 语义特征，输入输出通道一致，尺寸 1/8 256
        sp = F.interpolate(sp, cp.size()[2:], mode='bilinear', align_corners=True)

        # x_8
        cp = self.conv_8(cp)

        # 将 cp1x1 和 sp1x1 与 conv_offset 合并，以减少计算量
        cp1x1 = self.cp1x1(cp)
        sp1x1 = self.sp1x1(sp)

        conv_results = self.conv_offset(torch.cat([cp1x1, sp1x1], 1))

        # 重塑sp和cp以适应offset操作
        sp = sp.reshape(n * self.groups, -1, out_h, out_w)
        cp = cp.reshape(n * self.groups, -1, out_h, out_w)

        offset_l = conv_results[:, 0:self.groups * 2, :, :].reshape(n * self.groups, -1, out_h, out_w)
        offset_h = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(n * self.groups, -1, out_h, out_w)

        # 创建网格用于grid_sample
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(sp).to(sp.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n * self.groups, 1, 1, 1).type_as(sp).to(sp.device)

        # 通过网格偏移来调整sp和cp
        grid_l = grid + offset_l.permute(0, 2, 3, 1) / norm
        grid_h = grid + offset_h.permute(0, 2, 3, 1) / norm

        # 使用grid_sample进行采样
        cp = F.grid_sample(cp, grid_l, align_corners=True)
        sp = F.grid_sample(sp, grid_h, align_corners=True)

        # 恢复原始尺寸
        cp = cp.reshape(n, -1, out_h, out_w)
        sp = sp.reshape(n, -1, out_h, out_w)

        # 计算注意力机制
        att = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        
        # att = torch.sigmoid(conv_results[:, self.groups * 4:, :, :])  # 使用 Sigmoid 激活函数

        # 最终合并特征
        sp = sp * att[:, 0:1, :, :] + cp * att[:, 1:2, :, :]

        return sp


class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0,
            bias = bias, **kwargs)
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvBNReLU2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride = 1,
                 padding=1,
                 bias = False,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2 if padding else 0, bias = bias,**kwargs)

        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    # print("x before mean and max:", x.shape)
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    # mean_value = mean_value.unsqueeze(0)
    # print("mean max:", mean_value.shape, max_value.shape)

    if use_concat:
        res = torch.at([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            # print(xi.shape)
            res.extend(avg_max_reduce_channel_helper(xi, False))
        # print("res:\n",)
        # for it in res:
        #     print(it.shape)
        return torch.cat(res, dim=1)


class UMF(nn.Module):
    def __init__(self, in_ch, ksize=3, resize_mode='nearest'):
        super().__init__()
        self.conv_y = ConvBNReLU(
            in_ch, in_ch, kernel_size=ksize, padding=ksize // 2, bias=False)
        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU2(
                6, 6, kernel_size=3, padding=1, bias=False),
            ConvBN(
                6, 3, kernel_size=3, padding=1, bias=False))
        self.resize_mode = resize_mode

    def prepare(self, x, y, z):
        x = self.prepare_x(x, y, z)
        y = self.prepare_y(x, y, z)
        z = self.prepare_z(x, y, z)
        return x, y, z

    def prepare_x(self, x, y, z):
        return x

    def prepare_y(self, x, y, z):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        y_up = self.conv_y(y_up)
        return y_up

    def prepare_z(self, x, y, z):
        return z

    def fuse(self, x, y, z):
        atten = avg_max_reduce_channel([x, y, z])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        w1, w2, w3 = torch.split(atten, 1, 1)

        out = x * w1 + y * w2 + z * w3
        return out

    def forward(self, x, y, z):
        x, y, z = self.prepare(x, y, z)
        out = self.fuse(x, y, z)
        return out


class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add
    

class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        # Conv-BN-ReLU block for fusion
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        
        # Use conv-bn instead of 2-layer MLP for fp16 support with TensorRT 7.2.3.4
        self.conv = nn.Conv2d(out_chan,
                              out_chan,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

        # Initialize weights for the convolutional layers
        self.init_weight()

    def forward(self, fsp, fcp, fthird):
        # Concatenate the three input tensors along the channel dimension
        fcat = torch.cat([fsp, fcp, fthird], dim=1)  # Concatenate along channels (dim=1)
        
        # Apply the convolution block (Conv-BN-ReLU)
        feat = self.convblk(fcat)
        
        # Calculate attention as a global average pool
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        
        # Apply convolution and batch normalization to the attention map
        atten = self.conv(atten)
        atten = self.bn(atten)
        
        # Apply sigmoid activation to get the attention weights
        atten = atten.sigmoid()

        # Multiply the features by the attention weights
        feat_atten = torch.mul(feat, atten)
        
        # Add the attention-modulated features back to the original features
        feat_out = feat_atten + feat
        
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, bias=False)                  
                                )

        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)


def rotate_every_two(x):
    # 对输入张量的最后一个维度，每隔两个元素分组，将其中一个分量进行反转
    x1 = x[:, :, :, :, ::2]  # 选取每隔两个元素的第一个
    x2 = x[:, :, :, :, 1::2]  # 选取每隔两个元素的第二个
    x = torch.stack([-x2, x1], dim=-1)  # 交换顺序并按维度堆叠
    return x.flatten(-2)  # 将最后两个维度合并成一个维度


def theta_shift(x, sin, cos):
    # 应用旋转变换，基于正弦和余弦调整输入张量
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):
    # 深度可分离卷积类
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # 定义深度卷积：每个通道独立处理
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c) 输入张量的形状为 (批次, 高度, 宽度, 通道数)
        '''
        x = x.permute(0, 3, 1, 2)  # 转置为 (b c h w) 以适配卷积操作
        x = self.conv(x)  # 应用深度卷积
        x = x.permute(0, 2, 3, 1)  # 转置回原始形状 (b h w c)
        return x


class RetNetRelPos2d(nn.Module):
    # RetNet 的二维相对位置编码模块
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        initial_value: 初始衰减值
        heads_range: 衰减范围
        '''
        super().__init__()
        # 初始化角度参数，用于相对位置的正弦和余弦编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 重复以获得正弦和余弦部分
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        # 初始化衰减参数
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
    def generate_2d_decay(self, H: int, W: int):
        '''
        生成二维的衰减掩码
        H, W: 张量的高度和宽度
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])  # 创建网格
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 重塑为二维坐标
        mask = grid[:, None, :] - grid[None, :, :]  # 计算相对位置差
        mask = (mask.abs()).sum(dim=-1)  # 计算曼哈顿距离
        mask = mask * self.decay[:, None, None]  # 加权衰减
        return mask
    def generate_1d_decay(self, l: int):
        '''
        生成一维的衰减掩码
        l: 序列长度
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # 计算相对位置差
        mask = mask.abs()  # 取绝对值
        mask = mask * self.decay[:, None, None]  # 加权衰减
        return mask
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w) 输入张量的高度和宽度
        activate_recurrent: 是否启用循环计算
        chunkwise_recurrent: 是否启用分块循环计算
        '''
        if activate_recurrent:
            # 循环模式：基于总长度生成正弦和余弦
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            # 分块模式：对每块单独生成正弦和余弦，以及一维掩码
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])
            retention_rel_pos = ((sin, cos), (mask_h, mask_w))
        else:
            # 普通模式：生成正弦、余弦和二维掩码
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(slen[0], slen[1], -1)
            mask = self.generate_2d_decay(slen[0], slen[1])
            retention_rel_pos = ((sin, cos), mask)
        return retention_rel_pos


class VisionRetentionChunk(nn.Module):
    # Vision Retention 模块，用于实现视觉特征的注意力机制
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5  # 缩放因子，用于规范化
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 查询投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 键值投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)  # 值投影层
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)  # 局部增强模块
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)  # 输出投影层
        self.reset_parameters()
    def reset_parameters(self):
        # 初始化权重参数
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c) 输入张量
        rel_pos: 相对位置编码
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos
        # 计算查询、键和值向量
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)  # 计算局部增强特征
        # 对查询和键进行缩放和旋转变换
        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        # 计算宽度方向上的注意力
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)
        # 计算高度方向上的注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)
        # 恢复输出形状并添加局部增强特征
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output


class ManhattanBag(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # 三分支特征转换
        self.p_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.i_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.d_conv = nn.Conv2d(in_channels, out_channels, 1)

        # 曼哈顿注意力模块
        self.pos_encoder = RetNetRelPos2d(
            embed_dim=out_channels,
            num_heads=num_heads,
            initial_value=5,
            heads_range=3
        )
        self.retention = VisionRetentionChunk(
            embed_dim=out_channels,
            num_heads=num_heads
        )

        # 融合层
        self.fusion = nn.Sequential(
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

        # 注意力门控
        self.att_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, p, i, d):
        """
        p: 细节特征 (B,C,H,W)
        i: 语义特征 (B,C,H,W)
        d: 边界特征 (B,C,H,W)
        """
        # 1. 特征投影
        p = self.p_conv(p.permute(0, 2, 3, 1).permute(0, 3, 1, 2))  # 保持BCHW格式
        i = self.i_conv(i.permute(0, 2, 3, 1).permute(0, 3, 1, 2))
        d = self.d_conv(d.permute(0, 2, 3, 1).permute(0, 3, 1, 2))

        # 2. 转换为BHWC格式
        p_bhwc = p.permute(0, 2, 3, 1)
        i_bhwc = i.permute(0, 2, 3, 1)
        d_bhwc = d.permute(0, 2, 3, 1)

        # 3. 曼哈顿注意力处理
        _, h, w, _ = p_bhwc.shape
        rel_pos = self.pos_encoder((h, w), chunkwise_recurrent=True)

        # 三分支注意力增强
        p_att = self.retention(p_bhwc, rel_pos).permute(0, 3, 1, 2)
        i_att = self.retention(i_bhwc, rel_pos).permute(0, 3, 1, 2)
        d_att = self.retention(d_bhwc, rel_pos).permute(0, 3, 1, 2)

        # 4. 自适应融合
        edge_att = self.att_gate(d_att)  # 边缘注意力权重
        fused = edge_att * p_att + (1 - edge_att) * i_att + d_att  # 三分支融合

        # 5. 最终输出
        return self.fusion(fused)


if __name__ == '__main__':

    
    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()
    
    out = net(x,y)

