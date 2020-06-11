from utils import *
from PIL import Image

if __name__ == '__main__':
    patch_pth = 'patch_sample.png'
    patch = Image.open(patch_pth).convert('RGB')
    [W, H] = patch.size
    # keypoint location
    x = W//2
    y = H//2
    descriptor = patch_R2D2(patch, x, y)
    print('keypoint: (%i, %i)\tpatch size: %i x %i'%(x,y,H,W))
    print('descriptor:\n', descriptor)