import random
import functools
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# 主要功能是对输入的图像列表随机应用一种图像增强变换,包括水平翻转、垂直翻转和3种角度的旋转(90°、180°、270°)
class RandomFlipOrRotate(object):
    funcs = [TF.hflip, 
             TF.vflip, 
             functools.partial(TF.rotate, angle=90), 
             functools.partial(TF.rotate, angle=180), 
             functools.partial(TF.rotate, angle=270)]
    def __call__(self, imgs: list):
        rand = random.randint(0, 4)
        
        for i in range(len(imgs)):
            img = imgs[i]
            imgs[i] =self.funcs[rand](img)

        return imgs


