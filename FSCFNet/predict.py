import time
from PIL import Image
from models.sseg.mynet import *
from torchvision import transforms
import tqdm
import os
import numpy as np
import torch.nn as nn
import torch


def load_model(model, model_path):
    # 加载模型权重
    checkpoint = torch.load(model_path)

    # 处理 "module." 前缀
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        new_key = key.replace('module.', '')  # 移除 "module." 前缀
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

def get_test_img(ids, dir, suffix):
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

        yield [img_n, id, img_b, img_nir, img_rgb]



def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))

def predict_img(net,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=True,
                save_img=True):
    net.eval()

    dir_img = '/media/estar/Data/HWJ/improv-cloudnet/test_nir/'

    """Returns a list of the ids in the directory"""
    id_image = get_ids(dir_img)

    test_img = get_test_img(id_image, dir_img, '.TIF')

    val_dice = 0
    oa, recall, precision, f1, iou_score = 0, 0, 0, 0, 0

    # 使用tqdm创建进度条
    progress_bar = tqdm.tqdm(test_img, unit="image")
    for i, b in enumerate(progress_bar):
        progress_bar.set_description(f"Processing image {b[1]}")
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
            output_img = net(img, img_rgb,img_nir, img_b)[0]
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
            result_img.save(save_path + b[1] + '.TIF')

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='/media/estar/Data/HWJ/FSCFNet/checkpoints_landsat8/saved_models/mynet/2024090614_best_acc.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', default='image1.TIF',
                        help='filenames of input images')
    parser.add_argument('--output', '-o', default='output.jpg', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--gpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--save_img', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=True)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        # 最小概率值考虑掩模像素为白色
                        default=0.5)
    parser.add_argument("--path", type=str, default='/media/estar/Data/HWJ/MCDNet-main/checkpoints_landsat8/saved_models/mynet/2024090315_best_acc.pth', help="path")

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    # 从数组array转成Image
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args() #得到输入的选项设置的值
    in_files = args.input
    out_files = get_output_filenames(args)
    save_path = 'result_landsat8/'

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # net = torch.load(args.model)
    net = Cloudnet(n_channels=4, n_classes=1)

    print("Loading model {}".format(args.model))


    if args.gpu:
        print("Using CUDA version of the net, prepare your GPU !")
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        net = load_model(net, args.path)
        net.cuda()
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    time1 = time.time()
    predict_img(net=net,
                out_threshold=args.mask_threshold,
                use_dense_crf=args.no_crf,
                save_img=args.save_img)
    print(time.time() - time1)