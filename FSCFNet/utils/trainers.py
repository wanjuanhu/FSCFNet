import sys, math
import time, datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.config import Options

from .loss import BoundaryLoss, mmIoULoss,structure_loss,linear_annealing
import os
from utils.loss import muti_loss_fusion





class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def to_cropped_mask(ids, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img = Image.open("/media/estar/Data/HWJ/improv-cloudnet/train_gt/" + id + suffix)
        im = np.array(img)
        im = im/255
        yield im

def to_cropped_imgs_nir(ids, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_nir = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_nir/train/' + 'nir_' + id[3:] + suffix)
        img_nir = np.array(img_nir)
        img_nir = img_nir[np.newaxis, ...]

        yield img_nir

def to_cropped_imgs_b(ids, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_b = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_blue/' + 'blue_' + id[3:] + suffix)
        img_b = np.array(img_b)

        img_b = img_b[np.newaxis, ...]
        yield img_b


def to_cropped_imgs(ids, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_r = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_red/' + 'red_' + id[3:] + suffix)
        img_g = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_green/' + 'green_' + id[3:] + suffix)
        img_b = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_blue/' 'blue_' + id[3:] + suffix)
        img_nir = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_nir/train/' + 'nir_' + id[3:] + suffix)
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_nir = np.array(img_nir)
        img_r = img_r[np.newaxis, ...]
        img_g = img_g[np.newaxis, ...]
        img_b = img_b[np.newaxis, ...]
        img_nir = img_nir[np.newaxis, ...]
        img_n = np.stack((img_r, img_g, img_b, img_nir), axis=0)
        img_n = np.concatenate((img_n),axis=0)

        yield img_n


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def to_cropped_imgs_rgb(ids, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        img_r = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_red/' + 'red_' + id[3:] + suffix)
        img_g = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_green/' + 'green_' + id[3:] + suffix)
        img_b = Image.open('/media/estar/Data/HWJ/improv-cloudnet/train_blue/' + 'blue_' + id[3:] + suffix)
        img_r = np.array(img_r)
        img_g = np.array(img_g)
        img_b = np.array(img_b)
        img_r = img_r[np.newaxis, ...]
        img_g = img_g[np.newaxis, ...]
        img_b = img_b[np.newaxis, ...]
        img_rgb = np.stack((img_r,img_g,img_b),axis=0)
        img_rgb = np.concatenate((img_rgb), axis=0)

        yield img_rgb


def split_train_val(dataset, val_percent=0.1):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    return {'val': dataset[-n:]}

def get_imgs_and_masks(args, ids, suffix):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, suffix)
    #img_rgb = to_cropped_imgs_rgb(ids, dir_img, '.TIF')
    img_nir = to_cropped_imgs_nir(ids, suffix)
    img_b = to_cropped_imgs_b(ids, suffix)
    # imgs_normalized = map(normalize, imgs)
    # imgs_normalized = map(hwc_to_chw, imgs_normalized)
    masks = to_cropped_mask(ids, suffix)
    img_rgb = to_cropped_imgs_rgb(ids, suffix)
    if args.model_name == "mcdnet":
        img_rgb = to_cropped_removal(ids, suffix)


    return zip(imgs, img_rgb, img_nir, img_b, masks)

def to_cropped_removal(ids, suffix):
    for id in ids:
        img_removal = Image.open("/media/estar/Data/HWJ/MCDNet-main/thin_cloud_landsat8/" + 'rgb_' + id[3:] + suffix)
        im = np.array(img_removal)
        im = im / 255
        im = np.array(im).astype(np.float32)
        im = torch.from_numpy(im)
        im = im.permute(2, 0, 1)
        im = np.array(im)
        yield im




def get_val_data(args, dataset):
    if dataset == "landsat8":
        id_image = get_ids("/media/estar/Data/HWJ/improv-cloudnet/train_gt/")
        val_dataset = split_train_val(id_image, 0.1)
        val = get_imgs_and_masks(args, val_dataset['val'], ".TIF")
        return val










class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_img, self.next_rgb,self.next_n, self.next_b,\
            self.next_label = next(self.loader)
        except StopIteration:
            self.next_img = None
            self.next_rgb = None
            self.next_n = None
            self.next_b = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.cuda(non_blocking=True)
            self.next_rgb = self.next_rgb.cuda(non_blocking=True)
            self.next_n = self.next_n.cuda(non_blocking=True)
            self.next_b = self.next_b.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        imgs, img_rgb, img_n, img_b, label = self.next_img, self.next_rgb, self.next_n, self.next_b, self.next_label
        if imgs is not None:
            imgs.record_stream(torch.cuda.current_stream())
        if img_rgb is not None:
            img_rgb.record_stream(torch.cuda.current_stream())
        if img_n is not None:
            img_n.record_stream(torch.cuda.current_stream())
        if img_b is not None:
            img_b.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return imgs, img_rgb, img_n, img_b, label


class BaseTrainer(object):

    def __init__(self, args, model, device) -> None:
        self.model_name = args.model_name
        self.model = model
        self.device = device
        self.args = args
        self.cel = nn.BCELoss().to(device)
        self.boundary_loss = BoundaryLoss()
        self.mmiou_loss = mmIoULoss(n_classes=self.args.num_classes)
        self._init_optimizer()
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size

    def _init_optimizer(self):
        self.optimizer, self.lr_scheduler = None, None
        if self.model_name == 'cdnetv2':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, 
                                             momentum=0.9, weight_decay=0.0005)
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        if self.args.start_epoch>1 and self.lr_scheduler!=None:
            self.lr_scheduler.step(self.args.start_epoch-1)

    def cal_loss(self, img, rgb, nir, blue, label):
        if self.model_name == 'cdnetv2':
            pred, pred_aux = self.model(img)
            loss_pred = self.cel(pred, label)
            loss_aux = self.cel(pred_aux, label)
            loss = loss_pred + loss_aux
        elif self.model_name == 'mcdnet':
            predict = self.model(img, rgb)
            loss = self.cel(predict, label)
        elif self.model_name == "mynet" or self.model_name == "mynet01" or self.model_name == "mynet02" or self.model_name == "mynet03" or self.model_name == "mynet04":
            mask1, mask2, mask3, mask4 = self.model(img, rgb, nir, blue)
            loss1 = self.cel(mask1, label)
            loss2 = self.cel(mask2, label)
            loss3 = self.cel(mask3, label)
            loss4 = self.cel(mask4, label)
            loss = loss1 + loss2 + loss3 + loss4
        elif self.model_name == "boundarynet":
            d0, d1, d2, d3, d4, d5, d6, d7 = self.model(img)
            loss0, loss = muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, label)
        elif self.model_name == "HRcloudNet":
            predict = self.model(img)
            loss = self.cel(predict['out'], label)
        elif self.model_name == "MINet":
            lat_loss, out, out3, out2, out1 = self.model(nir, rgb)
            loss4 = structure_loss(out, label)
            loss3 = structure_loss(out3, label)
            loss2 = structure_loss(out2, label)
            loss1 = structure_loss(out1, label)


        else:
            predict = self.model(img)
            loss = self.cel(predict, label)

        if self.model_name == "MINet":
            return loss1, loss2, loss3, loss4, lat_loss

        else:
            return loss

    def train(self, epoch, train_loader):
        if self.model_name == 'MINet':
            loss_record4, loss_record3, loss_record2, loss_record1, lat_loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()


        prev_time = time.time()
        prefetcher = data_prefetcher(train_loader)
        img, rgb, nir, blue, label = prefetcher.next()
        i = 1
        while img is not None:
            self.optimizer.zero_grad()

            if self.model_name == 'MINet':
                loss1, loss2, loss3, loss4, lat_loss = self.cal_loss(img, rgb, nir, blue, label)
                anneal_reg = linear_annealing(0, 1, epoch, self.n_epochs)
                latent_loss = 0.1 * anneal_reg * lat_loss
                #loss = loss4 + loss3 + loss2 + loss1 + latent_loss
                loss = loss4 + loss3 + loss2 + loss1
            else:
                loss = self.cal_loss(img, rgb, nir, blue, label)
            loss.backward()
            self.optimizer.step()

            if self.model_name == 'MINet':
                loss_record4.update(loss4.data, self.batch_size)
                loss_record3.update(loss3.data, self.batch_size)
                loss_record2.update(loss2.data, self.batch_size)
                loss_record1.update(loss1.data, self.batch_size)
                #lat_loss_record.update(latent_loss.data, self.batch_size)

            # Determine approximate time left
            batches_done = (epoch - 1) * len(train_loader) + i
            batches_left = self.args.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
            prev_time = time.time()
            
            #  Log Progress
            sys.stdout.write(
                "\r[Epoch %03d/%d] [Batch %03d/%d] [Cross Entropy Loss: %7.4f] ETA: %8s"
                % (
                    epoch,
                    self.args.n_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    time_left,
                )
            )

            if self.model_name == 'MINet':
                # print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                #       '[lateral-4: {:.4f}], [lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [mi: {:,.4f}],ETA: {:8s}'.
                #       format(epoch, self.batch_size, i, len(train_loader),
                #              loss_record4.avg, loss_record3.avg, loss_record2.avg, loss_record1.avg,
                #              lat_loss_record.avg, str(time_left)))
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-4: {:.4f}], [lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], ETA: {:8s}'.
                      format(epoch, self.batch_size, i, len(train_loader),
                             loss_record4.avg, loss_record3.avg, loss_record2.avg, loss_record1.avg,
                             str(time_left)))

            i += 1
            img, rgb, nir, blue, label  = prefetcher.next()
        
        if self.lr_scheduler != None:
            self.lr_scheduler.step()



