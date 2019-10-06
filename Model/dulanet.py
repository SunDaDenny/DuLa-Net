import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .resnet import resnet18, resnet34, resnet50
from .e2p import E2P

sys.path.append('..')
import config as cf

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_relu(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))

class DulaNet_Branch(nn.Module):
    def __init__(self, backbone):
        super(DulaNet_Branch, self).__init__()

        bb_dict = {'resnet18':resnet18, 
                    'resnet34':resnet34, 
                    'resnet50':resnet50}

        self.encoder = bb_dict[backbone]()

        feat_dim = 512 if backbone != 'resnet50' else 2048

        self.decoder = nn.ModuleList([
            conv3x3_relu(feat_dim, 256),
            conv3x3_relu(256, 128),
            conv3x3_relu(128, 64),
            conv3x3_relu(64, 32),
            conv3x3_relu(32, 16),
        ])
        self.last = conv3x3(16, 1)

    def forward_get_feats(self, x):
        x = self.encoder(x)
        feats = [x]
        for conv in self.decoder:
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
            feats.append(x)
        out = self.last(x)
        return out, feats

    def forward_from_feats(self, x, feats):
        x = self.encoder(x)
        for i, conv in enumerate(self.decoder):
            x = x + feats[i]
            x = F.interpolate(x, scale_factor=(2,2), mode='nearest')
            x = conv(x)
        out = self.last(x)
        return out
    
class DuLaNet(nn.Module):
    def __init__(self, backbone):
        super(DuLaNet, self).__init__()

        self.model_equi = DulaNet_Branch(backbone)
        self.model_up = DulaNet_Branch(backbone)

        self.model_h = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.Dropout(inplace=True),
                    nn.Linear(256, 64),
                    nn.Dropout(inplace=True),
                    nn.Linear(64, 1)
                )
        
        self.e2p = E2P(cf.pano_size, cf.fp_size, cf.fp_fov)

        fuse_dim = [int((cf.pano_size[0]/32)*2**i) for i in range(6)]
        self.e2ps_f = [E2P((n, n*2), n, cf.fp_fov) for n in fuse_dim]

    def forward(self, pano_view):

        [up_view, down_view] = self.e2p(pano_view)

        fcmap, feats_equi = self.model_equi.forward_get_feats(pano_view)

        feats_fuse = []
        for i, feat in enumerate(feats_equi):
            [feat_up, _] = self.e2ps_f[i](feat)
            feats_fuse.append(feat_up * 0.6 * (1/3)**i)

        fpmap = self.model_up.forward_from_feats(up_view, feats_fuse)

        fpmap = torch.sigmoid(fpmap)
        fcmap = torch.sigmoid(fcmap)
        height = self.model_h(feats_equi[0].mean(1).view(-1, 512))

        return fpmap, fcmap, height

                
if __name__ == '__main__':
   
    model = DuLaNet(backbone='resnet50').cuda()
    batch = torch.ones(4, 3, 512, 1024).cuda()
    fpmap, fcmap, height = model(batch)
    print(fpmap.shape)
    print(fcmap.shape)
    print(height.shape)
