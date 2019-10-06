import os
import sys
import argparse

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms

import Layout
import Utils

import config as cf
from Model import DuLaNet, E2P

import postproc

parser = argparse.ArgumentParser(description='DuLa-Net inference scripts')

parser.add_argument('--backbone', default='resnet18',  
                    choices=['resnet18', 'resnet34', 'resnet50'], help='backbone network')
parser.add_argument('--ckpt',  default='./Model/ckpt/res18_realtor.pkl',  
                    help='path to the model ckpt file')

parser.add_argument('--input', type=str, help='input panorama image')
parser.add_argument('--output',  default='./output', type=str, help='output path')

parser.add_argument('--cpu', action='store_true', help='using cpu or not')
parser.add_argument('--seed', default=224, type=int, help='manual random seed')
parser.add_argument('--processes', default=8, type=int, help='processes number')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu
                        else 'cpu')  

def predict(model, input_path):
    print('predict')
    model.eval()

    img = Image.open(input_path).convert("RGB")

    trans = transforms.Compose([
            transforms.Resize((cf.pano_size)),
            transforms.ToTensor()
        ])
    color = torch.unsqueeze(trans(img), 0).to(device)

    [fp, fc, h] = model(color)

    e2p = E2P(cf.pano_size, cf.fp_size, cf.fp_fov)
    [fc_up, fc_down] = e2p(fc)

    [fp, fc_up, fc_down, h] = Utils.var2np([fp, fc_up, fc_down, h])
    fp_pts, fp_pred = postproc.run(fp, fc_up, fc_down, h)

    # Visualization 
    scene_pred = Layout.pts2scene(fp_pts, h)
    edge = Layout.genLayoutEdgeMap(scene_pred, [512 , 1024, 3], dilat=2, blur=0)

    img = img.resize((1024,512))
    img = np.array(img, np.float32) / 255
    vis = img * 0.5 + edge * 0.5

    vis = Image.fromarray(np.uint8(vis* 255))
    vis.save(os.path.splitext(input_path)[0] + "_vis.jpg")

    #Save output 3d layout as json
    Layout.saveSceneAsJson(os.path.splitext(input_path)[0] + "_res.json", scene_pred)

    return


def demo():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = DuLaNet(args.backbone).to(device)
    
    assert args.ckpt is not None, "need pretrained model"
    assert args.input, "need an input for prediction"

    #model.load_state_dict(torch.load(args.ckpt))
    model.load_state_dict(torch.load(args.ckpt, map_location=str(device)))

    predict(model, args.input)

if __name__ == '__main__':
    demo()






