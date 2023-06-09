
from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return DIM(args)

class DIM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DIM, self).__init__()

        self.scale = args.scale[0]

        # feature extractor part
        self.fe_conv1 = common.BasicBlock(conv, args.n_colors, 196, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv2 = common.BasicBlock(conv, 196, 166, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv3 = common.BasicBlock(conv, 166, 148, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv4 = common.BasicBlock(conv, 148, 133, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv5 = common.BasicBlock(conv, 133, 120, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv6 = common.BasicBlock(conv, 120, 108, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv7 = common.BasicBlock(conv, 108, 97, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv8 = common.BasicBlock(conv, 97, 86, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv9 = common.BasicBlock(conv, 86, 76, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv10 = common.BasicBlock(conv, 76, 66, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv11 = common.BasicBlock(conv, 66, 57, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv12 = common.BasicBlock(conv, 57, 48, kernel_size=3, bias=True, act=nn.PReLU())

        # reconstruction part
        self.re_a = common.BasicBlock(conv, 196 + 48, 64, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_b1 = common.BasicBlock(conv, 196 + 48, 32, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_b2 = common.BasicBlock(conv, 32, 32, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_u = common.Upsampler(conv, self.scale, 96, act=False)
        self.re_r = conv(96, args.n_colors, kernel_size=1)


    def forward(self, x):

        residual = F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        # feature extractor part
        fe_conv1 = self.fe_conv1(x)
        fe_conv2 = self.fe_conv2(fe_conv1)
        fe_conv3 = self.fe_conv3(fe_conv2)
        fe_conv4 = self.fe_conv4(fe_conv3)
        fe_conv5 = self.fe_conv5(fe_conv4)
        fe_conv6 = self.fe_conv6(fe_conv5)
        fe_conv7 = self.fe_conv7(fe_conv6)
        fe_conv8 = self.fe_conv8(fe_conv7)
        fe_conv9 = self.fe_conv9(fe_conv8)
        fe_conv10 = self.fe_conv10(fe_conv9)
        fe_conv11 = self.fe_conv11(fe_conv10)
        fe_conv12 = self.fe_conv12(fe_conv11)

        # reconstruction part
        feat = torch.cat((fe_conv1, fe_conv12), dim=1)
        re_a = self.re_a(feat)
        re_b1 = self.re_b1(feat)
        re_b2 = self.re_b2(re_b1)
        feat = torch.cat((re_a, re_b2), dim=1)
        re_u = self.re_u(feat)
        re_r = self.re_r(re_u)
        out = re_r + residual

        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
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

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
if __name__ == "__main__":
    from option import args
    import time
    import torch
    net = DIM(args).cuda()
    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()
    y = net(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    from thop import profile

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


