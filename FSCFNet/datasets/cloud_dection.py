import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from datasets.transform import RandomFlipOrRotate



# class ImageDataset(Dataset):
#     def __init__(self, args, mode="train", normalization=True):
#         self.args = args
#         self.mode = mode
#         self.normalization = normalization
#         self.cloudy_paths, self.dc_paths, self.label_paths = \
#             self.get_path_pairs([os.path.join(args.root, mode.title(), args.cloudy),
#                                  os.path.join(args.root, mode.title(), args.dc),
#                                  os.path.join(args.root, mode.title(), args.label)])
#         self.length = len(self.cloudy_paths)
#         self.RandomFlipOrRotate = RandomFlipOrRotate()
#
#     def get_path_pairs(self, paths):
#         first_names = os.listdir(paths[0])
#         common_names = list(set(first_names).intersection(
#             *[os.listdir(paths[i]) for i in range(1, len(paths))]
#         ))
#         common_names = sorted(common_names, key=lambda name: first_names.index(name))
#
#         common_names = [common_name for common_name in common_names if common_name.endswith(self.args.file_suffix)]
#         return ([os.path.join(path, name) for name in common_names] for path in paths)
#
#     def __getitem__(self, index):
#         i = index % self.length
#         cloudy = cv2.imread(self.cloudy_paths[i]).transpose(2, 0, 1)
#         label = cv2.imread(self.label_paths[i]).transpose(2, 0, 1)[0, ...][np.newaxis, ...]
#
#         cloudy = torch.from_numpy(cloudy).float().div(255)
#         dc = torch.from_numpy(dc).float().div(255)
#         label = torch.from_numpy(label).long()
#
#         if self.mode == 'train':
#             cloudy, dc, label = self.RandomFlipOrRotate([cloudy, dc, label])
#
#         if self.normalization:
#             cloudy = TF.normalize(cloudy, [0.5] * cloudy.size()[0], [0.5] * cloudy.size()[0])
#             dc = TF.normalize(dc, [0.5] * dc.size()[0], [0.5] * dc.size()[0])
#
#         label = torch.squeeze(label)
#         return cloudy, dc, label
#
#     def __len__(self):
#         return self.length


class ImageDataset2(Dataset):
    def __init__(self, args, mode="train", normalization=True):
        self.args = args
        self.mode = mode
        self.model = args.model_name
        self.normalization = normalization
        self.cloudy_paths = "/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/"
        self.label_paths = "/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/nir/"
        self.ids = self.get_ids(self.label_paths)
        self.length = len(self.ids)
        self.RandomFlipOrRotate = RandomFlipOrRotate()

    def get_ids(self, dir):
        ids = []
        count = 0
        for filename in os.listdir(dir):
            if filename.endswith(".png"):
                ids.append(os.path.splitext(filename)[0])
                count = count + 1
                if count >= 17888:
                    break
        return ids

    # 在这个方法中，他的实例化对象假定为p，可以用作p[key]取值，当实例化对象p[key]运算时，会调用类中的方法
    # 当调用train中的Dataloader的时候会自动进入此处进行遍历。
    def __getitem__(self, item):
        imgs = self.get_img(item)
        img_rgb = self.get_rgb(item)
        img_n = self.get_n(item)
        img_b = self.get_b(item)
        img_mask = self.get_mask(item)
        if self.model == "mcdnet":
            img_rgb = self.get_removal(item)

        return imgs, img_rgb, img_n, img_b, img_mask


    def get_removal(self,item):
        img_removal = Image.open("/media/estar/Data/HWJ/MCDNet-main/thin_cloud/" + self.ids[item] + ".png")
        im = np.array(img_removal)
        im = im / 255
        im = np.array(im).astype(np.float32)
        im = torch.from_numpy(im)
        im = im.permute(2, 0, 1)
        return im


    def get_mask(self, item):
        img = Image.open('/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/train_msk/' + self.ids[item] + ".png")
        im = np.array(img)
        im = im / 255
        im = im[np.newaxis, ...]
        im = np.array(im).astype(np.float32)
        im = torch.from_numpy(im)
        return im


    def get_rgb(self, item):
        img_r = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/red/" + self.ids[item] + ".png")
        img_g = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/green/" + self.ids[item] + ".png")
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/blue/" + self.ids[item] + ".png")
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)


        img_rgb = np.stack((img_r, img_g, img_b), axis=0)
        img_rgb = np.array(img_rgb).astype(np.float32)
        img_rgb = torch.from_numpy(img_rgb)

        return img_rgb

    def get_n(self,item):
        img_n = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/nir/" + self.ids[item] + ".png")
        img_n = np.array(img_n)
        img_n = img_n[np.newaxis, ...]
        img_n = np.array(img_n).astype(np.float32)
        img_n = torch.from_numpy(img_n)
        return img_n

    def get_b(self,item):
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/blue/" + self.ids[item] + ".png")
        img_b = np.array(img_b)
        img_b = img_b[np.newaxis, ...]
        img_b = np.array(img_b).astype(np.float32)
        img_b = torch.from_numpy(img_b)

        return img_b

    def get_img(self,item):
        img_r = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/red/" + self.ids[item] + ".png")
        img_g = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/green/" + self.ids[item] + ".png")
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/blue/" + self.ids[item] + ".png")
        img_n = Image.open("/media/estar/Data/HWJ/improv-cloudnet/modis/modis_train/nir/" + self.ids[item] + ".png")
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_n = np.array(img_n)


        img = np.stack((img_r, img_g, img_b, img_n), axis=0)
        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img)

        return img


    # 返回容器中元素的个数。
    def __len__(self):
        return self.length



class ImageDataset3(Dataset):
    def __init__(self, args, mode="train", normalization=True):
        self.args = args
        self.mode = mode
        self.normalization = normalization
        self.label_paths = "/media/estar/Data/HWJ/improv-cloudnet/train_gt/"
        self.ids = self.get_ids(self.label_paths)
        self.length = len(self.ids)
        self.RandomFlipOrRotate = RandomFlipOrRotate()

    def get_ids(self, dir):
        ids = []
        count = 0
        for filename in os.listdir(dir):
            if filename.endswith(".TIF"):
                ids.append(os.path.splitext(filename)[0])
                count += 1
                if count >= 7560:
                    break
        return ids

    # 在这个方法中，他的实例化对象假定为p，可以用作p[key]取值，当实例化对象p[key]运算时，会调用类中的方法
    # 当调用train中的Dataloader的时候会自动进入此处进行遍历。
    def __getitem__(self, item):
        imgs = self.get_img(item)
        img_rgb = self.get_rgb(item)
        img_n = self.get_n(item)
        img_b = self.get_b(item)
        img_mask = self.get_mask(item)
        if self.args.model_name == "mcdnet":
            img_rgb = self.get_removal(item)

        return imgs, img_rgb, img_n, img_b, img_mask

    def get_removal(self, item):
        img_removal = Image.open("/media/estar/Data/HWJ/MCDNet-main/thin_cloud_landsat8/" + 'rgb_' + self.ids[item][3:] + ".TIF")
        im = np.array(img_removal)
        im = im / 255
        im = np.array(im).astype(np.float32)
        im = torch.from_numpy(im)
        im = im.permute(2, 0, 1)
        return im


    def get_mask(self, item):
        img = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_gt/' + self.ids[item] + ".TIF")
        im = np.array(img)
        im = im / 255
        im = im[np.newaxis, ...]
        im = np.array(im).astype(np.float32)
        im = torch.from_numpy(im)
        return im


    def get_rgb(self, item):
        img_r = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_red/" + 'red_' + self.ids[item][3:] + ".TIF")
        img_g = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_green/" + 'green_' + self.ids[item][3:] + ".TIF")
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_blue/" + 'blue_' + self.ids[item][3:] + ".TIF")
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)


        img_rgb = np.stack((img_r, img_g, img_b), axis=0)
        img_rgb = np.array(img_rgb).astype(np.float32)
        img_rgb = torch.from_numpy(img_rgb)

        return img_rgb

    def get_n(self,item):
        img_n = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_nir/train/" + 'nir_' + self.ids[item][3:] + ".TIF")
        img_n = np.array(img_n)
        img_n = img_n[np.newaxis, ...]
        img_n = np.array(img_n).astype(np.float32)
        img_n = torch.from_numpy(img_n)
        return img_n

    def get_b(self,item):
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_blue/" + 'blue_' + self.ids[item][3:] + ".TIF")
        img_b = np.array(img_b)
        img_b = img_b[np.newaxis, ...]
        img_b = np.array(img_b).astype(np.float32)
        img_b = torch.from_numpy(img_b)

        return img_b

    def get_img(self,item):
        img_r = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_red/" + 'red_' + self.ids[item][3:] + ".TIF")
        img_g = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_green/" + 'green_' + self.ids[item][3:] + ".TIF")
        img_b = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_blue/" + 'blue_' + self.ids[item][3:] + ".TIF")
        img_n = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_nir/train/" + 'nir_' + self.ids[item][3:] + ".TIF")
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_n = np.array(img_n)


        img = np.stack((img_r, img_g, img_b, img_n), axis=0)
        img = np.array(img).astype(np.float32)
        img = torch.from_numpy(img)

        return img


    # 返回容器中元素的个数。
    def __len__(self):
        return self.length


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


