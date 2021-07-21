
import torch.nn as nn
import torch.nn.functional as F
import torch as t


class SimiHeads(nn.Module):
    
    def __init__(self, representation_size):
        super(SimiHeads, self).__init__()

        self.fc = nn.Linear(representation_size*representation_size, 1)
        # self.conv_shape = nn.Sequential(
        #         nn.Conv2d(channels, channels,  kernel_size=3, stride=1,padding=1),
        #         nn.BatchNorm2d(channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channels, 4,  kernel_size=3, stride=1,padding=1),
        #         )
        # self.conv1=nn.Sequential(
        #         nn.Conv2d(channels, channels,  kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(channels),
        #         nn.ReLU(inplace=True),
        #         )

        # self.conv2=nn.Sequential(
        #         nn.Conv2d(channels, channels,  kernel_size=3, stride=1),
        #         nn.BatchNorm2d(channels),
        #         nn.ReLU(inplace=True),
        #         )

        # for modules in [self.conv1,self.conv2]:
        #     for l in modules.modules():
        #         if isinstance(l, nn.Conv2d):
        #             t.nn.init.normal_(l.weight, std=0.01)
        #             t.nn.init.constant_(l.bias, 0)


    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        num = kernel.size(0)
        channel = kernel.size(1)
        a = kernel.size(2)
        # padding = int((a-1)/2)
        # can't use view
        x = x.reshape(-1, num*channel, x.size(3), x.size(4))
        kernel = kernel.view(-1, num*channel, kernel.size(2)*kernel.size(3))
        kernel = t.sigmoid(self.fc(kernel))
        kernel = kernel.reshape(kernel.size(0), kernel.size(1), kernel.size(2), -1).expand(x.size(0), x.size(1), x.size(2), x.size(3))
        out = t.mul(x,kernel)+x
        out = out.view(x.size(0), num, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):
        # x=self.conv1(x)
        # z=self.conv2(z)
        # x: b * n * d * w * h
        # n: n * d * w * h
        x1 = x.view(x.size(0), -1, x.size(1), x.size(2), x.size(3)).expand(x.size(0), z.size(0), x.size(1), x.size(2), x.size(3))
        res=self.xcorr_depthwise(x1,z)
        # shape_pred=self.conv_shape(res)

        return res

