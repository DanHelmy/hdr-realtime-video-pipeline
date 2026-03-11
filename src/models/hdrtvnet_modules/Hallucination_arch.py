import torch
import torch.nn as nn


def init_weights(net, init_type="normal"):
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(f"initialization method [{init_type}] is not implemented")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif "Linear" in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, norm=True, act=True):
    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding)
    n = nn.BatchNorm2d(out_nc, affine=True) if norm else None
    a = nn.ReLU(inplace=True) if act else None
    layers = [c, n, a]
    return nn.Sequential(*[x for x in layers if x is not None])


def up_block(in_nc, out_nc, sf=2, kernel_size=3, stride=1):
    c = nn.Conv2d(in_nc, out_nc * sf ** 2, kernel_size=kernel_size, stride=stride, padding=1)
    s = nn.PixelShuffle(sf)
    a = nn.ReLU(inplace=True)
    return nn.Sequential(c, s, a)


class Hallucination_Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, sf=2):
        super().__init__()

        self.maxpool = nn.MaxPool2d(sf)

        self.conv1 = conv_block(in_nc, nf)
        self.conv2 = conv_block(nf, 2 * nf)

        self.conv3_1 = conv_block(2 * nf, 4 * nf)
        self.conv3_2 = conv_block(4 * nf, 4 * nf)

        self.conv4_1 = conv_block(4 * nf, 8 * nf)
        self.conv4_2 = conv_block(8 * nf, 8 * nf)

        self.conv5_1 = conv_block(8 * nf, 8 * nf)
        self.conv5_2 = conv_block(8 * nf, 8 * nf)

        self.conv_code1 = conv_block(8 * nf, 8 * nf)
        self.conv_code2 = conv_block(8 * nf, 8 * nf)

        self.Up_conv1 = up_block(8 * nf, 8 * nf, sf=2)
        self.conv6 = nn.Conv2d(16 * nf, 8 * nf, 1, 1)

        self.Up_conv2 = up_block(8 * nf, 8 * nf, sf=2)
        self.conv7 = nn.Conv2d(16 * nf, 4 * nf, 1, 1)

        self.Up_conv3 = up_block(4 * nf, 4 * nf, sf=2)
        self.conv8 = nn.Conv2d(8 * nf, 2 * nf, 1, 1)

        self.Up_conv4 = up_block(2 * nf, 2 * nf, sf=2)
        self.conv9 = nn.Conv2d(4 * nf, nf, 1, 1)

        self.Up_conv5 = up_block(nf, nf, sf=2)
        self.conv10 = nn.Conv2d(2 * nf, out_nc, 1, 1)

        self.conv_last = nn.Conv2d(2 * out_nc, out_nc, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        img = x[0]
        mask = x[1]

        conv1_out = self.conv1(img)

        conv2_out = self.maxpool(conv1_out)
        conv2_out = self.conv2(conv2_out)

        conv3_out = self.maxpool(self.conv3_1(conv2_out))
        conv3_out = self.conv3_2(conv3_out)

        conv4_out = self.maxpool(self.conv4_1(conv3_out))
        conv4_out = self.conv4_2(conv4_out)

        conv5_out = self.maxpool(self.conv5_1(conv4_out))
        conv5_out = self.conv5_2(conv5_out)

        conv_code = self.maxpool(self.conv_code1(conv5_out))
        conv_code = self.conv_code2(conv_code)

        conv6_out = torch.cat((self.Up_conv1(conv_code), conv5_out), 1)
        conv6_out = self.conv6(conv6_out)

        conv7_out = torch.cat((self.Up_conv2(conv6_out), conv4_out), 1)
        conv7_out = self.conv7(conv7_out)

        conv8_out = torch.cat((self.Up_conv3(conv7_out), conv3_out), 1)
        conv8_out = self.conv8(conv8_out)

        conv9_out = torch.cat((self.Up_conv4(conv8_out), conv2_out), 1)
        conv9_out = self.conv9(conv9_out)

        conv10_out = torch.cat((self.Up_conv5(conv9_out), conv1_out), 1)
        conv10_out = self.conv10(conv10_out)

        out = torch.cat((conv10_out, img), 1)
        out = self.conv_last(out)

        out = mask * out + img
        return out
