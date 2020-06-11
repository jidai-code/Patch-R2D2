import torchvision.transforms as tvf
import torch
from PIL import Image
from patchnet import *
import math
import numpy as np
from imageio import imwrite
from matplotlib import pyplot as plt
from glob import glob
import os

RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])

''' 
default params
'''
k_TOPK = 5000
k_SCALE_F = 2**0.25
k_MIN_SIZE = 256
k_MAX_SIZE = 1024
k_MIN_SCALE = 0
k_MAX_SCALE = 1
k_RELIAB_THRD = 0.7
k_REPEAT_THRD = 0.7

def load_network(model_pth): 
    print("Net: %s"%(model_pth))
    checkpoint = torch.load(model_pth)
    net = eval(checkpoint['net'])
    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    net = net.eval().cuda()
    return net

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=k_RELIAB_THRD, rep_thr=k_REPEAT_THRD):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaksextract_keypoints
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]

def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

def extract_keypoints(img_pth, net):
    detector = NonMaxSuppression()
    img = Image.open(img_pth).convert('RGB')
    [W, H] = img.size
    img = norm_RGB(img)[None] 
    img = img.cuda()
    [xys, desc, scores] = extract_multiscale(
        net,
        img,
        detector,
        scale_f   = k_SCALE_F,
        verbose = False)
    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-k_TOPK or None:]
    keypoints = xys[idxs]
    descriptors = desc[idxs]
    scores = scores[idxs]
    return [keypoints, descriptors, scores, [W,H]]
    
def extract_single_keypoint(
    img_pth,
    net,
    keypoint,
    scale_f = k_SCALE_F,
    min_scale = k_MIN_SCALE,
    max_scale = k_MAX_SCALE,
    min_size  = k_MIN_SIZE,
    max_size  = k_MAX_SIZE):
    detector = NonMaxSuppression()
    img = Image.open(img_pth).convert('RGB')
    [W, H] = img.size
    img = norm_RGB(img)[None] 
    img = img.cuda()
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    s = 1
    for _ in range(round(math.log(keypoint[2]/32,k_SCALE_F))):
        s /= scale_f
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    nh, nw = img.shape[2:]
    y = int(round(keypoint[1]*s))
    x = int(round(keypoint[0]*s))

    w_size = 25
    # print("====")
    # print(img.shape)
    # print(y-w_size,y+w_size+1)
    # print(x-w_size,x+w_size+1)
    img = img[:,:,max(y-w_size,0):y+w_size+1,max(x-w_size,0):x+w_size+1]
    # print(img.shape)
    # if not (img.shape[2]==w_size*2+1 and img.shape[3]==w_size*2+1):
    #     return None
    with torch.no_grad():
        res = net(imgs=[img])
    
    # print(img.shape)
    # get output and reliability map
    descriptors = res['descriptors'][0]
    reliability = res['reliability'][0]
    repeatability = res['repeatability'][0]
    
    # y = int(round(keypoint[1]*s))
    # x = int(round(keypoint[0]*s))
    x0 = w_size + min(x-w_size,0)
    y0 = w_size + min(y-w_size,0)
    c = reliability[0,0,y0,x0]
    q = repeatability[0,0,y0,x0]
    d = descriptors[0,:,y0,x0].t()
    n = d.shape[0]

    # accumulate multiple scales
    x *= W/nw
    y *= H/nh
    s = 32/s

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    score = c*q
    keypoint = [x,y,s]
    # keypoint = keypoints.cpu().numpy()
    descriptor = d.cpu().numpy()
    score = score.cpu().numpy()
    return [keypoint, descriptor, score]

def get_patch(
    img_pth,
    keypoint,
    scale_f = k_SCALE_F,
    min_scale = k_MIN_SCALE,
    max_scale = k_MAX_SCALE,
    min_size  = k_MIN_SIZE,
    max_size  = k_MAX_SIZE):
    
    img = Image.open(img_pth).convert('RGB')
    [W, H] = img.size
    img = norm_RGB(img)[None] 
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    s = 1
    for _ in range(round(math.log(keypoint[2]/32,k_SCALE_F))):
        s /= scale_f
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    nh, nw = img.shape[2:]
    y = int(round(keypoint[1]*s))
    x = int(round(keypoint[0]*s))

    w_size = 25
    img = img[:,:,max(y-w_size,0):y+w_size+1,max(x-w_size,0):x+w_size+1]
    x0 = w_size + min(x-w_size,0)
    y0 = w_size + min(y-w_size,0)
    patch = img.numpy()
    return [patch, x0, y0]

def patch_R2D2(
    patch,
    x,
    y,
    net=None):

    if not torch.is_tensor(patch):
        patch = norm_RGB(patch)[None] 
        patch = patch.cuda()
    
    if net is None:
        model_pth = 'r2d2_WASF_N16.pt'
        net = load_network(model_pth)
    
    with torch.no_grad():
        res = net(imgs=[patch])
    descriptors = res['descriptors'][0]
    d = descriptors[0,:,y,x].t()
    return d.cpu().numpy()