
import torch
import torch.nn as nn
import time
def make_model(args, parent=False):
    return LGCNET(args)

class LGCNET(nn.Module):
    def __init__(self, args, nfeats = 32):
        super(LGCNET, self).__init__()
        self.conv1 = nn.Conv2d(args.n_colors, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(nfeats*3, nfeats*2, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv7 = nn.Conv2d(nfeats*2, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu =  nn.ReLU()

    def forward(self, x):
        residual = x
        im1 = self.relu(self.conv1(x))
        im2 = self.relu(self.conv2(im1))
        im3 = self.relu(self.conv3(im2))
        im4 = self.relu(self.conv4(im3))
        im5 = self.relu(self.conv5(im4))
        out = self.relu(self.conv6(torch.cat((im3, im4, im5), dim = 1)))
        out = self.conv7(out) + residual
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
if __name__ == "__main__":
    from option import args
    import psutil
    import os
    net = LGCNET(args).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.5fM" % (total / 1e6))
    x = torch.rand(1, 3, 64*4, 64*4)
    x = x.cuda()
    torch.cuda.reset_max_memory_allocated()
    y = net(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()
    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))

    from thop import profile

    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1000000.0))
