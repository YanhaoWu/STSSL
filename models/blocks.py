import torch.nn as nn
import MinkowskiEngine as ME

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels, batch_nor=False, pix_level=False):
        nn.Module.__init__(self)

        if batch_nor == True:
            self.projection_head = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels),
            )        
        print("if you used the batch normalize, it is", batch_nor)    
        self.dropout = ME.MinkowskiDropout(p=0.4)
        self.pix_level = pix_level # 这玩意没用上
        # if pix_level==False:
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x, pix_deal=False):           # pix_deal需要给成默认为False，否则K头进入的时候就没法glob了
        # from input points dropout some (increase randomness)
        x = self.dropout(x)

        if pix_deal==False:        # 如果进行点级别的对应计算，则有一个头是不需要进行池化计算cluster的特征的
            x = self.glob_pool(x)

        # project the max pooled features
        out = self.projection_head(x.F)

        return out

class PredictionHead(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, hidden_channels=256):
        nn.Module.__init__(self)

        self.liner1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.liner2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):

        # project the max pooled features
        z = self.liner1(x)
        z = self.bn1(z)
        z = self.relu(z)
        out = self.liner2(z)

        return out



class SegmentationClassifierHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=26):
        nn.Module.__init__(self)

        self.fc = nn.Sequential(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        return self.fc(x.F)

class ClassifierHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=40):
        nn.Module.__init__(self)

        self.fc = ME.MinkowskiLinear(in_channels, out_channels, bias=True)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x):
        return self.fc(self.glob_pool(x))
