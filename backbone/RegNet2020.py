#import pycls.core.net as net
import torch.nn as nn
import numpy as np

def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.num_preds = 150
        self.num_modes = 3
        self.future_len = 50
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print("Warning, this fc layer is only for lyft")
        self.fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=w_in, out_features=4096, bias=True),
            # nn.ReLU(),
            nn.Linear(4096, out_features=self.num_preds + self.num_modes)
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)        
        x = self.fc(x)

        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    # @staticmethod
    # def complexity(cx, w_in, nc):
    #     cx["h"], cx["w"] = 1, 1
    #     cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
    #     return cx


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
    #     err_str = "Vanilla block does not support bm, gw, and se_r options"
    #     assert bm is None and gw is None and se_r is None, err_str
    #     cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     return cx


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        self.a_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride):
    #     cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     return cx


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Basic transform does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(ResBasicBlock, self).__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
    #     err_str = "Basic transform does not support bm, gw, and se_r options"
    #     assert bm is None and gw is None and se_r is None, err_str
    #     proj_block = (w_in != w_out) or (stride != 1)
    #     if proj_block:
    #         h, w = cx["h"], cx["w"]
    #         cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
    #         cx = net.complexity_batchnorm2d(cx, w_out)
    #         cx["h"], cx["w"] = h, w  # parallel branch
    #     cx = BasicTransform.complexity(cx, w_in, w_out, stride)
    #     return cx


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    # @staticmethod
    # def complexity(cx, w_in, w_se):
    #     h, w = cx["h"], cx["w"]
    #     cx["h"], cx["w"] = 1, 1
    #     cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
    #     cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
    #     cx["h"], cx["w"] = h, w
    #     return cx


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride, bm, gw, se_r):
    #     w_b = int(round(w_out * bm))
    #     g = w_b // gw
    #     cx = net.complexity_conv2d(cx, w_in, w_b, 1, 1, 0)
    #     cx = net.complexity_batchnorm2d(cx, w_b)
    #     cx = net.complexity_conv2d(cx, w_b, w_b, 3, stride, 1, g)
    #     cx = net.complexity_batchnorm2d(cx, w_b)
    #     if se_r:
    #         w_se = int(round(w_in * se_r))
    #         cx = SE.complexity(cx, w_b, w_se)
    #     cx = net.complexity_conv2d(cx, w_b, w_out, 1, 1, 0)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     return cx


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
    #     proj_block = (w_in != w_out) or (stride != 1)
    #     if proj_block:
    #         h, w = cx["h"], cx["w"]
    #         cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
    #         cx = net.complexity_batchnorm2d(cx, w_out)
    #         cx["h"], cx["w"] = h, w  # parallel branch
    #     cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, bm, gw, se_r)
    #     return cx


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out):
    #     cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     return cx


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out):
    #     cx = net.complexity_conv2d(cx, w_in, w_out, 7, 2, 3)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     cx = net.complexity_maxpool2d(cx, 3, 2, 1)
    #     return cx


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out):
    #     cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
    #     cx = net.complexity_batchnorm2d(cx, w_out)
    #     return cx


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    # @staticmethod
    # def complexity(cx, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
    #     for i in range(d):
    #         b_stride = stride if i == 0 else 1
    #         b_w_in = w_in if i == 0 else w_out
    #         cx = block_fun.complexity(cx, b_w_in, w_out, b_stride, bm, gw, se_r)
    #     return cx


class AnyNet(nn.Module):
    """AnyNet model."""

    @staticmethod
    def get_args(cfg):
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "ds": cfg.ANYNET.DEPTHS,
            "ws": cfg.ANYNET.WIDTHS,
            "ss": cfg.ANYNET.STRIDES,
            "bms": cfg.ANYNET.BOT_MULS,
            "gws": cfg.ANYNET.GROUP_WS,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else None,
            "nc": cfg.CLASS_NUM,
        }

    def __init__(self, cfg, logger, **kwargs):
        super(AnyNet, self).__init__()
        kwargs = self.get_args(cfg) if not kwargs else kwargs
        self._construct(**kwargs)
        #self.apply(net.init_weights)

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    # @staticmethod
    # def complexity(cx, **kwargs):
    #     """Computes model complexity. If you alter the model, make sure to update."""
    #     kwargs = AnyNet.get_args() if not kwargs else kwargs
    #     return AnyNet._complexity(cx, **kwargs)

    # @staticmethod
    # def _complexity(cx, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
    #     bms = bms if bms else [None for _d in ds]
    #     gws = gws if gws else [None for _d in ds]
    #     stage_params = list(zip(ds, ws, ss, bms, gws))
    #     stem_fun = get_stem_fun(stem_type)
    #     cx = stem_fun.complexity(cx, 3, stem_w)
    #     block_fun = get_block_fun(block_type)
    #     prev_w = stem_w
    #     for d, w, s, bm, gw in stage_params:
    #         cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, bm, gw, se_r)
    #         prev_w = w
    #     cx = AnyHead.complexity(cx, prev_w, nc)
    #     return cx


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_args(cfg):
        """Convert RegNet to AnyNet parameter format."""
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
        ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        # Convert to per stage format
        s_ws, s_ds = get_stages_from_blocks(ws, ws)
        # Use the same gw, bm and ss for each stage
        s_gs = [cfg.REGNET.GROUP_W for _ in range(num_stages)]
        s_bs = [cfg.REGNET.BOT_MUL for _ in range(num_stages)]
        s_ss = [cfg.REGNET.STRIDE for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "ds": s_ds,
            "ws": s_ws,
            "ss": s_ss,
            "bms": s_bs,
            "gws": s_gs,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else None,
            "nc": cfg.CLASS_NUM,
        }

    def __init__(self, cfg=None, logger=None):
        kwargs = RegNet.get_args(cfg)
        super(RegNet, self).__init__(cfg, logger, **kwargs)

    # @staticmethod
    # def complexity(cx, **kwargs):
    #     """Computes model complexity. If you alter the model, make sure to update."""
    #     kwargs = RegNet.get_args() if not kwargs else kwargs
    #     return AnyNet.complexity(cx, **kwargs)