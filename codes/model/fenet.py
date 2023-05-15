import torch
import torch.nn as nn

import torch.nn.functional as F

def make_model(args, parent=False):
    return FENet(upscale_factor=args.scale[0])
def mean_channels(x):
    assert (x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])


class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)


# lightweight lattice block
class LLBlock(nn.Module):
    def __init__(self, num_fea):
        super(LLBlock, self).__init__()
        self.channel1 = num_fea // 2
        self.channel2 = num_fea - self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )

        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A * x1), dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B * x2), dim=1)

        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)
        out = self.fuse(c)
        return out


# attention fuse
class AF(nn.Module):
    def __init__(self, num_fea):
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea)
        self.CA2 = CALayer(num_fea)
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)

    def forward(self, x1, x2):
        x1 = self.CA1(x1) * x1
        x2 = self.CA2(x2) * x2
        return self.fuse(torch.cat((x1, x2), dim=1))


# Feature enhancement block
class FEB(nn.Module):
    def __init__(self, num_fea):
        super(FEB, self).__init__()
        self.CB1 = LLBlock(num_fea)
        self.CB2 = LLBlock(num_fea)
        self.CB3 = LLBlock(num_fea)
        self.AF1 = AF(num_fea)
        self.AF2 = AF(num_fea)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.CB2(x1)
        x3 = self.CB3(x2)
        f1 = self.AF1(x3, x2)
        f2 = self.AF2(f1, x1)
        return x + f2


class RB(nn.Module):
    def __init__(self, num_fea):
        super(RB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea * 2, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_fea * 2, num_fea, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x) + x


class BFModule(nn.Module):
    def __init__(self, num_fea):
        super(BFModule, self).__init__()
        self.conv4 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv3 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.fuse43 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.fuse32 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv1 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x_list):
        H4 = self.act(self.conv4(x_list[3]))
        H3_half = self.act(self.conv3(x_list[2]))
        H3 = self.fuse43(torch.cat([H4, H3_half], dim=1))
        H2_half = self.act(self.conv2(x_list[1]))
        H2 = self.fuse32(torch.cat([H3, H2_half], dim=1))
        H1_half = self.act(self.conv1(x_list[0]))
        H1 = torch.cat([H2, H1_half], dim=1)

        return H1


# super lightweight feature enhancement network (FENet)
class FENet(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=48, out_channels=3, num_LBs=4):
        super(FENet, self).__init__()
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)
        self.num_LBs = num_LBs
        self.upscale_factor = upscale_factor
        # feature extraction
        self.fea_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_fea, 3, 1, 1),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )

        # LBlocks
        LBs = []
        for i in range(num_LBs):
            LBs.append(FEB(num_fea))
        self.LBs = nn.ModuleList(LBs)

        # BFModule
        self.BFM = BFModule(num_fea)

        # Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.Conv2d(num_fea, out_channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        bi = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        # feature extraction
        fea = self.fea_conv(x)

        # LBlocks
        outs = []
        temp = fea
        for i in range(self.num_LBs):
            temp = self.LBs[i](temp)
            outs.append(temp)

        # BFM
        H = self.BFM(outs)

        # reconstruct
        out = self.upsample(H + fea)

        return out + bi


def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    from torchstat import stat
    import time
    net = FENet(upscale_factor=4).cuda()
    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()
    y = net(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    from thop import profile
    x = torch.rand(1, 3, 64, 64).cuda()
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))
    net = net.cuda()
    x = x.cuda()
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()
    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))



