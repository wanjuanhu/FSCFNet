import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from datasets.cloud_dection import ImageDataset3
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import Recall
from torchmetrics.classification import Precision
from torchmetrics.classification import Specificity
from torchmetrics.classification import F1Score
from torchmetrics.classification import CohenKappa
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from image_stack_landsat8 import ImageStack
from utils.config import Options


def get_test_img(args, ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        # img_r = Image.open(dir + 'test_384_384_red/' + 'red_' + id[3:] + suffix)
        img_r = Image.open('/media/estar/Data/HWJ/improv-cloudnet/test_red/' + "red_" + id[4:] + suffix)
        # img_g = Image.open(dir + 'test_384_384_green/' + 'green_' + id[3:] + suffix)
        img_g = Image.open('/media/estar/Data/HWJ/improv-cloudnet/test_green/' + "green_" + id[4:] + suffix)
        # img_b = Image.open(dir + 'test_384_384_blue/' + 'blue_' + id[3:] + suffix)
        img_b = Image.open('/media/estar/Data/HWJ/improv-cloudnet/test_blue/' + "blue_" + id[4:] + suffix)
        img_nir = Image.open('/media/estar/Data/HWJ/improv-cloudnet/test_nir/' + id + suffix)
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_nir = np.array(img_nir)
        img_r = img_r[np.newaxis, ...]
        img_g = img_g[np.newaxis, ...]
        img_b = img_b[np.newaxis, ...]
        img_nir = img_nir[np.newaxis, ...]
        img_n = np.stack((img_r, img_g, img_b, img_nir), axis=0)
        img_n = np.concatenate((img_n), axis=0)
        # img_rgb_normalized = normalize(img_rgb)
        img_rgb = np.stack((img_r, img_g, img_b), axis=0)
        img_rgb = np.concatenate((img_rgb), axis=0)
        if args.model_name == "mcdnet":
            img_removal = Image.open("/media/estar/Data/HWJ/MCDNet-main/thin_cloud_landsat8_test/" + "rgb_" + id[4:] + suffix)
            im = np.array(img_removal)
            im = im / 255
            im = np.array(im).astype(np.float32)
            img_rgb = torch.from_numpy(im)
            img_rgb = img_rgb.permute(2, 0, 1)
            img_rgb = np.array(img_rgb)


        yield [img_n, id, img_b, img_nir, img_rgb]



def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

from .trainers import data_prefetcher
def mask_to_image(mask):
    # 从数组array转成Image
    return Image.fromarray((mask * 255).astype(np.uint8))

class Evaluator2(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix)[1] / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Pixel_Precision_Class(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        precision = np.nanmean(precision)
        return precision

    def Pixel_Recall_Class(self):
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        recall = np.nanmean(recall)
        return recall

    def Pixel_JaccardIndex_Class(self):
        jaccard_index = np.diag(self.confusion_matrix)[1] / \
                        (np.sum(self.confusion_matrix, axis=0)[1] + self.confusion_matrix[1, 0])
        return jaccard_index

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # 消除边界
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def get_test_img2(args, ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        # img_r = Image.open(dir + 'test_384_384_red/' + 'red_' + id[3:] + suffix)
        img_r = Image.open('/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/red/' + id + suffix)
        # img_g = Image.open(dir + 'test_384_384_green/' + 'green_' + id[3:] + suffix)
        img_g = Image.open('/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/green/' + id + suffix)
        # img_b = Image.open(dir + 'test_384_384_blue/' + 'blue_' + id[3:] + suffix)
        img_b = Image.open('/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/blue/' + id + suffix)
        img_nir = Image.open('/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/nir/' + id + suffix)
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_nir = np.array(img_nir)
        img_r = img_r[np.newaxis, ...]
        img_g = img_g[np.newaxis, ...]
        img_b = img_b[np.newaxis, ...]
        img_nir = img_nir[np.newaxis, ...]
        img_n = np.stack((img_r, img_g, img_b, img_nir), axis=0)
        img_n = np.concatenate((img_n), axis=0)
        # img_rgb_normalized = normalize(img_rgb)
        img_rgb = np.stack((img_r, img_g, img_b), axis=0)
        img_rgb = np.concatenate((img_rgb), axis=0)
        if args.model_name == "mcdnet":
            img_removal = Image.open("/media/estar/Data/HWJ/MCDNet-main/thin_cloud_test/" + id + ".png")
            im = np.array(img_removal)
            im = im / 255
            im = np.array(im).astype(np.float32)
            img_rgb = torch.from_numpy(im)
            img_rgb = img_rgb.permute(2, 0, 1)
        yield [img_n, id, img_b, img_nir, img_rgb]

def do_predict2(args, net, savepath, dataset):
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    torch.backends.cudnn.enabled = False
    net = net.eval()
    out_threshold = 0.5
    use_gpu = True
    save_img = True

    if dataset == "modis":
        dir_img = '/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/nir/'

        """Returns a list of the ids in the directory"""
        id_image = get_ids(dir_img)

        test_img = get_test_img2(args, id_image, dir_img, '.png')  # 返回图像三维数组,图像名称


    val_dice = 0
    oa, recall, precision, f1, iou_score = 0, 0, 0, 0, 0

    for i, b in enumerate(test_img):
        img = np.array(b[0]).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        img_b = np.array(b[2]).astype(np.float32)
        img_b = torch.from_numpy(img_b).unsqueeze(0)
        img_nir = np.array(b[3]).astype(np.float32)
        img_nir = torch.from_numpy(img_nir).unsqueeze(0)
        img_rgb = np.array(b[4]).astype(np.float32)
        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        if use_gpu:
            img = img.cuda()
            img_b = img_b.cuda()
            img_nir = img_nir.cuda()
            img_rgb = img_rgb.cuda()

        with torch.no_grad():
            output_img = net(img)[0]
            probs = output_img.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                ]
            )

            probs = tf(probs.cpu())

            probs_np = probs.squeeze().cpu().numpy()

        result_img = mask_to_image(probs_np > out_threshold)
        if save_img:
            result_img.save(savepath + b[1] + '.png')


    Gt_Dir = '/media/estar/Data/HWJ/improv-cloudnet/modis/modis_test/test_msk/'
    evaluator = Evaluator2(2)
    evaluator.reset()
    # /home/ices/lixian/dataset/Test/Entire_scene_gts/edited_corrected_gts_LC08_L1TP_003052_20160120_20170405_01_T1.TIF
    for filename in tqdm(os.listdir(savepath)):  # tqdm函数用于创建一个进度条，可以在循环中显示处理的进度。
        for file in os.listdir(Gt_Dir):  # 返回Gt_Dir目录下的文件列表，并将每个文件储在变量file中
            judge = filename  # 获取filename中的第一部分，不包括文件扩展名，并且在两侧添加下划字符，储存在judge中
            if judge in file:
                gt_filename = file
                break
        pre_file_path = os.path.join(savepath, filename)
        gt_file_path = os.path.join(Gt_Dir, file)
        assert os.path.isfile(pre_file_path)
        assert os.path.isfile(gt_file_path)
        pred = Image.open(pre_file_path).convert('1')
        target = Image.open(gt_file_path).convert('1')
        # pred.save('real.png','PNG')
        # w = np.ceil((pred.size[0] - target.size[0]) / 2)  #计算预测图像宽度和目标图像宽度之间的差值，ceil表示向上取整 /2
        # h = np.ceil((pred.size[1] - target.size[1]) / 2)  #计算预测图像高度和目标图像之间的差值，ceil表示向上取整  /2
        # pred = pred.crop((w, h, w + target.size[0], h + target.size[1]))
        # pred = pred.resize(target.size, Image.ANTIALIAS)
        # pred.save('pred.png','PNG')
        # target.save('target.png',"PNG")
        pred = np.asarray(pred).astype(int)
        # pred[pred>0] = 1
        target = np.asarray(target).astype(int)

        # assert 1==2
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    precision = evaluator.Pixel_Precision_Class()
    recall = evaluator.Pixel_Recall_Class()
    jaccard_index = evaluator.Pixel_JaccardIndex_Class()
    confusion_matrix = evaluator._generate_matrix(target, pred)
    print(confusion_matrix)
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {},precision:{},recall:{},jaccard_index:{}"
          .format(Acc, Acc_class, mIoU, FWIoU, precision, recall, jaccard_index))
    return Acc, Acc_class, mIoU, FWIoU, precision, recall, jaccard_index



def apply_color_map(num_classes, imgs):
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        cmap[i] = np.array([(i/(num_classes-1))*255]*3)

    out = []
    imgs = imgs.cpu().numpy()
    for i in range(len(imgs)):
        img = imgs[i]
        img = Image.fromarray(img.astype(np.uint8), mode="P")
        img.putpalette(cmap)
        img = img.convert('RGB')
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        out.append(img)
    out = np.concatenate(out, axis=0)
    return torch.from_numpy(out)

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range=None):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    t.mul_(255).add_(0.5).clamp_(0, 255)

def sample_images(args, dataloader, model, epoch):
    """Saves a generated sample from the validation set"""
    with torch.no_grad():
        (img, rgb, nir, blue, label) = next(dataloader)
        predict = get_pred(args, img, rgb, nir, blue, label)
        predict = torch.argmax(predict, dim=1, keepdim=True)
        
        label = apply_color_map(args.num_classes, label).to(img.device)
        predict = apply_color_map(args.num_classes, predict.squeeze()).to(img.device)
        norm_range(img), norm_range()


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix)[1] / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Pixel_Precision_Class(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        precision = np.nanmean(precision)
        return precision

    def Pixel_Recall_Class(self):
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        recall = np.nanmean(recall)
        return recall

    def Pixel_JaccardIndex_Class(self):
        jaccard_index = np.diag(self.confusion_matrix)[1] / \
                        (np.sum(self.confusion_matrix, axis=0)[1] + self.confusion_matrix[1, 0])
        return jaccard_index

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # 消除边界
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def get_pred(args, model, img, rgb, nir, blue):
    if args.model_name == 'mcdnet':
        predict = model(img, rgb)
    elif args.model_name == 'cdnetv2':
        predict, _ = model(img)
    elif args.model_name == 'mynet':
        output, mask2, mask3, mask4 = model(img, rgb, nir, blue)
        predict = output
    else:
        predict = model(img)
    return predict


def save_results(evl_path, metrics, indicators, epoch):
    try:
        metrics_df = pd.read_excel(evl_path, sheet_name=f'Sheet1')
    except:
        metrics_df = pd.DataFrame(columns=['Epoch'] + indicators)
    row = metrics_df.shape[0]
    metrics_df.loc[row, 'Epoch'] = epoch
    metrics_df.loc[row, indicators] = metrics

    try:
        if os.path.exists(evl_path):
            ew = pd.ExcelWriter(evl_path, mode='a', if_sheet_exists='replace', engine='openpyxl')
        else:
            ew = pd.ExcelWriter(evl_path)
        metrics_df.to_excel(ew, index=False, sheet_name=f'Sheet1')
        ew.close()
    except Exception as e:
        print(e)



