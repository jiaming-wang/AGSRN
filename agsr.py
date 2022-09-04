import torch
import torch.nn as nn
from model import common
# import common
from torch.autograd import Variable
# from model.common

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.scale = 4
        num_channels = 3
        self.phase = 2
        n_blocks = 16
        n_feats = 16
        kernel_size = 3

        act = nn.ReLU(True)
        conv=common.default_conv
        # self.upsample = nn.Upsample(scale_factor=self.scale,
        #                             mode='bicubic', align_corners=False)

        #stage 1
        self.head_1 = conv(num_channels, n_feats, kernel_size)

        self.down_1 = [
            common.DownBlock( 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down_1 = nn.ModuleList(self.down_1)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks_1 = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks_1.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), num_channels, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), num_channels, kernel_size)
            )
        self.tail_1 = nn.ModuleList(tail)

        #stage 2
        self.head_2 = conv(num_channels, n_feats, kernel_size)
        self.head_22 = conv(48, n_feats, kernel_size)

        self.down_2 = [
            common.DownBlock( 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down_2 = nn.ModuleList(self.down_2)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks_2 = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks_2.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), num_channels, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), num_channels, kernel_size)
            )
        self.tail_2 = nn.ModuleList(tail)

        self.sam12 = SAM(32)

    def forward(self, x):
        # upsample x to target sr size
        # x = self.upsample(x)
        bic = x

        #stage 1
        # preprocess
        x = self.head_1(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down_1[idx](x)

        # up phases
        sr = self.tail_1[0](x)

        # results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks_1[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            if idx == self.phase-1:
                sam_feature, out_1, att = self.sam12(x, bic)
            # output sr imgs
            sr = self.tail_1[idx + 1](x)
            # results.append(sr)
        sr_1 = out_1

        #stage 2
        x = self.head_2(sr_1)
        x = self.head_22(torch.cat([x, sam_feature], dim=1))

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down_2[idx](x)

        # up phases
        sr = self.tail_2[0](x)

        # results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks_1[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # if idx == self.phase-1:
            #     print(idx)
            # output sr imgs
            sr = self.tail_2[idx + 1](x)

            # results.append(sr)
        sr_2 = sr_1 + sr
      
        return sr_1, sr_2

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = -self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img, x2

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


