import os
import time
import torch
from torch.utils.data import DataLoader

from utils.config import Options
from models.model_zoo import get_model
from utils.trainers import BaseTrainer, get_val_data
from datasets.cloud_dection import ImageDataset2, ForeverDataIterator, ImageDataset3
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = 'cpu'



def evaluation(pred, target):
    eps = 0.0001
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred_oa = np.where(pred < 0.5, 0, 1)
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / (np.sum(union) + eps)
    oa = np.sum(np.equal(pred_oa, target)) / (target.shape[1] * target.shape[2])
    pred_tp = np.where(pred < 0.5, 0.5, 1)
    tp = np.sum(np.equal(pred_tp, target))
    recall = (tp + eps) / (np.sum(np.equal(target, 1)) + eps)
    precision = (tp + eps) / (np.sum(np.equal(pred, 1)) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return oa, recall, precision, f1, iou_score


def eval_net(args, net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    oa, recall, precision, f1, iou_score= 0, 0, 0, 0, 0
    for i, b in enumerate(dataset):
        img = b[0].astype(np.float32)
        img_rgb = b[1].astype(np.float32)
        img_nir = b[2].astype(np.float32)
        img_b = b[3].astype(np.float32)
        true_mask = b[4].astype(np.float32)



        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        img_nir = torch.from_numpy(img_nir).unsqueeze(0)
        img_b = torch.from_numpy(img_b).unsqueeze(0)
        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()
            img_nir = img_nir.cuda()
            img_b = img_b.cuda()
            img_rgb = img_rgb.cuda()

        if args.model_name == 'MINet':
            mask_pred = net(img_nir,img_rgb)[1]
            mask_pred = (mask_pred > 0.5).float()
            #mask_pred = F.upsample(mask_pred, size=true_mask.shape, mode='bilinear', align_corners=False)
            mask_pred = mask_pred.sigmoid()

        else:
            mask_pred = net(img)[0]


        mask_pred = (mask_pred > 0.5).float()


        oa += evaluation(mask_pred, true_mask)[0].item()
        recall += evaluation(mask_pred, true_mask)[1].item()
        precision += evaluation(mask_pred, true_mask)[2].item()
        f1 += evaluation(mask_pred, true_mask)[3].item()
        iou_score += evaluation(mask_pred, true_mask)[4].item()
    return iou_score / (i + 1), oa / (i + 1), recall / (i + 1), precision / (i + 1), f1 / (i + 1)




def save_model(model, args, name):
    torch.save(
        {'model': model.state_dict()},
        os.path.join('checkpoints_MINet', "compare_models", "MINet_noMI", f"{args.save_name}/{args.time}_{name}.pth")
        #os.path.join('checkpoints_MINet', "compare_models", f"{args.save_name}/{args.time}_{name}.pth")
    )


def main(args):
    # Initialize MODEL
    model = get_model(args, device)
    model = nn.DataParallel(model.cuda(), device_ids=[0])

    if args.checkpoint != '0':
        # Load pretrained models
        checkpoint = torch.load(os.path.join(args.root, f"saved_models/{args.save_name}/{args.checkpoint}.pth"))
        model.load_state_dict(checkpoint['model'])

    trainer = BaseTrainer(args, model, device)

    if args.dataset == "modis":
        train_loader = DataLoader(
            ImageDataset2(args, mode="train", normalization=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
            pin_memory=True
        )


    # Configure dataloaders
    if args.dataset == "landsat8":
        train_loader = DataLoader(
            ImageDataset3(args, mode="train", normalization=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
            pin_memory=True
        )


    #  Training
    acc = 0
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        model.train()
        trainer.train(epoch, train_loader)


        if args.dataset == "landsat8":
            val_data = get_val_data(args, dataset="landsat8")
            model.eval()
            iou, oa, recall, precision, f1 = eval_net(args, model, val_data, True)
            print('mIoU: {0}, Overall Accuracy:{1}, Recall: {2}, Precision: {3}, F-measure: {4}'
                  .format(iou, oa, recall, precision, f1))
            if oa > acc:
                acc = oa
                save_model(model, args, name=f'best_acc')

        if args.dataset == "modis":
            if epoch == 50 or epoch == 25 or epoch == 75 or epoch == 100:
                save_model(model, args, name=f'No_{epoch}')





        # If at sample interval evaluation




if __name__ == '__main__':
    # time.sleep(2 * 60 * 60)
    args = Options(model_name='mynet').parse(save_args=True) #1111111111111111111

    main(args)