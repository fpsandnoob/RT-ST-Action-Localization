import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import apex
from torch.autograd import Variable

from layers import *
from models.PWCNet import pwc_dc_net

VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v2',
}


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_c(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(BasicRFB_c, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch4 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch5 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )

        self.branch6 = nn.Sequential(
            BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=7, dilation=7, relu=False)
        )
        self.ConvLinear = BasicConv(7 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)

        out = torch.cat((x0, x1, x2, x3, x4, x5, x6), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class RFBNet(nn.Module):

    def __init__(self, phase, size, base, base_flo, extras, head, num_classes, flow_path="./pwc_net.pth.tar",
                 fusion=False, sw=True):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.fusion = fusion
        self.sw = sw

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only RFB300 and RFB512 are supported!")
            return

        self.priorbox = PriorBox(VOC_300)
        self.priors = Variable(self.priorbox.forward()).cuda()
        self.flow = pwc_dc_net(flow_path)
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3

        self.reduce = BasicConv(512, 256, kernel_size=1, stride=1)
        self.up_reduce = BasicConv(1024, 256, kernel_size=1, stride=1)

        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1).cuda()
        if fusion:
            self.base_flo = nn.ModuleList(base_flo)
            self.reduce_flo = BasicConv(512, 256, kernel_size=1, stride=1)
            self.up_reduce_flo = BasicConv(1024, 256, kernel_size=1, stride=1)

            self.Norm_flo = BasicRFB_a(512, 512, stride=1, scale=1.0)
            self.extras_flo = nn.ModuleList(extras)
            self.depthwise_flo = BasicConv(1024, 512, kernel_size=1, stride=1)
            self.pointwise_conv = BasicConv(4, 3, kernel_size=1)
        elif sw:
            self.pointwise_conv = BasicConv(2, 3, kernel_size=1)

    @staticmethod
    def flow_correct(flo):
        if len(flo) == 5:
            flo = flo[0]
            flo = F.upsample(flo, size=[300, 300], mode='bilinear')
            flo *= 300 / float(300)
        else:
            flo = F.upsample(flo, size=[300, 300], mode='bilinear')
            flo *= 300 / float(300)
        return flo

    def forward(self, data_x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        if self.fusion or self.sw:
            sources_flow = list()
        sources = list()
        loc = list()
        conf = list()

        x_0 = data_x[:, 0, ...]
        x_1 = data_x[:, 1, ...]
        x_2 = data_x[:, 2, ...]
        x_ = data_x[:, 3, ...]

        # if self.fusion or self.sw:
        #     f_0 = torch.cat([F.upsample_bilinear(x_0, size=(320, 320)), F.upsample_bilinear(x_1, size=(320, 320))], 1)
        #     f_1 = torch.cat([F.upsample_bilinear(x_1, size=(320, 320)), F.upsample_bilinear(x_2, size=(320, 320))], 1)
        #
        #     f_0 = self.flow(f_0)
        #     f_1 = self.flow(f_1)
        #
        #     f_0 = self.flow_correct(f_0)
        #     f_1 = self.flow_correct(f_1)
        #     flo = torch.cat([f_0, f_1], 1) * 20.0

        if self.fusion or self.sw:
            f = torch.cat([F.upsample(x_0, size=(320, 320), mode='bilinear', align_corners=False),
                           F.upsample(x_2, size=(320, 320), mode='bilinear', align_corners=False)], 1)
            f = self.flow(f)
            f = self.flow_correct(f)
            flo = f * 20.0

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_ = self.base[k](x_)

        s1 = self.reduce(x_)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x_ = self.base[k](x_)
        s2 = self.up_reduce(x_)
        s2 = F.upsample(s2, scale_factor=2, mode='bilinear', align_corners=False)
        # s2 = F.upsample_bilinear()
        s = torch.cat((s1, s2), 1)

        ss = self.Norm(s)
        sources.append(ss)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_ = v(x_)
            if k < self.indicator or k % 2 == 0:
                sources.append(x_)

        if self.fusion:
            # flo = self.pointwise_conv(flo)
            for k in range(23):
                flo = self.base_flo[k](flo)

            s1 = self.reduce_flo(flo)

            # apply vgg up to fc7
            for k in range(23, len(self.base_flo)):
                flo = self.base_flo[k](flo)
            s2 = self.up_reduce_flo(flo)
            s2 = F.upsample(s2, scale_factor=2, mode='bilinear', align_corners=True)
            # s2 = F.upsample_bilinear()
            s = torch.cat((s1, s2), 1)

            ss = self.Norm_flo(s)
            sources_flow.append(ss)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras_flo):
                flo = v(flo)
                if k < self.indicator or k % 2 == 0:
                    sources_flow.append(flo)

            _sources = list()
            for rgb, flow in zip(sources, sources_flow):
                _sources.append(torch.cat([rgb, flow], 1))
            sources = _sources
        elif self.sw:
            flo = self.pointwise_conv(flo)
            for k in range(23):
                flo = self.base[k](flo)

            s1 = self.reduce(flo)

            # apply vgg up to fc7
            for k in range(23, len(self.base)):
                flo = self.base[k](flo)
            s2 = self.up_reduce(flo)
            s2 = F.upsample(s2, scale_factor=2, mode='bilinear', align_corners=True)
            # s2 = F.upsample_bilinear()
            s = torch.cat((s1, s2), 1)

            ss = self.Norm(s)
            sources_flow.append(ss)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                flo = v(flo)
                if k < self.indicator or k % 2 == 0:
                    sources_flow.append(flo)

            _sources = list()
            for rgb, flow in zip(sources, sources_flow):
                _sources.append(torch.cat([rgb, flow], 1))
            sources = _sources


        # apply multibox head to source layers
        if self.fusion or self.sw:
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # print([o.size() for o in loc])

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256:
                    layers += [BasicRFB_c(in_channels, cfg[k + 1], stride=2, scale=1.0)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k + 1], stride=2, scale=1.0)]
            else:
                layers += [BasicRFB(in_channels, v, scale=1.0)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=4, stride=1, padding=1)]
    elif size == 300:
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
        layers += [BasicConv(256, 128, kernel_size=1, stride=1)]
        layers += [BasicConv(128, 256, kernel_size=3, stride=1)]
    else:
        print("Error: Sorry only RFB300 and RFB512 are supported!")
        return
    return layers


extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256, 'S', 256],
}


def multibox(size, vgg, vgg_flow, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers_ = []
    conf_layers_ = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(1024,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1024,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels * 2,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels * 2,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFB300 and RFB512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k % 2 == 0:
            loc_layers += [nn.Conv2d(v.out_channels * 2, cfg[i]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels * 2, cfg[i]
                                      * num_classes, kernel_size=3, padding=1)]
            i += 1
    return vgg, vgg_flow, extra_layers, (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFB300 and RFB512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3), vgg(base[str(size)], 2),
                                         add_extras(size, extras[str(size)], 1024),
                                         mbox[str(size)], num_classes), num_classes)
