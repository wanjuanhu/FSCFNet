# -*- coding: UTF-8 -*-
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
class ImageStack(object):
    def __init__(self,img_dir):

        self.IMAGE_DIR = img_dir   #将传入的 img_dir 赋值给 IMAGE_DIR 属性
        self.files = os.listdir(self.IMAGE_DIR)  #通过 os.listdir() 函数获取 IMAGE_DIR 中的所有文件列表，将其赋值给 files 属性。
        self.IMAGE_SIZE = 384  #将图片的大小设为 384，并将其赋值给 IMAGE_SIZE 属性。
        self.landSet_name_train = ['002053','002054','011002','011247','029040','032029','034034','035034',
                             '039034','044010','045026','047023','059014','061017',
                                   '063016','064014','064017','066017']

        self.landSet_name_test = ['035035', '063013', '035029', '032037', '066014', '029041', '032035', '029032',
                             '064012', '034037', '034029', '003052', '064015', '039035', '018008', '029044',
                             '034033', '032030', '050024', '063012']
        time_patches_eachImage = {}
        num_patches_eachImage = {}
        shape_paches_eachImage = {}

        '''如果 file 中包含 name，则更新 num_patches_eachImage 字典中 name 对应的值加 1，
        否则将 name 加入到 time_patches_eachImage 字典中，值为 file 文件名中的倒数第 4 个和倒数第 3 个字段，用 _ 连接起来。'''
        for file in self.files:
            for name in self.landSet_name_test:
                if '_'+name+'_' in file:
                    if name in num_patches_eachImage.keys():
                        num_patches_eachImage[name] += 1
                    else:
                        num_patches_eachImage[name] = 1

                    if name not in time_patches_eachImage.keys():
                        time_patches_eachImage[name] = file.split('_')[-4]+'_'+ file.split('_')[-3]
        # 将 num_patches_eachImage 字典中键为 '064012' 的值设为 529。
        num_patches_eachImage['064012'] = 529
        for key in num_patches_eachImage.keys():  # 遍历 num_patches_eachImage 字典中的键。
            value = str(num_patches_eachImage[key])  # 将 num_patches_eachImage 字典中当前键对应的值转化为字符串，赋值给变量 value。
            for file in self.files:
                if key in file and '_'+value+'_' in file:
                    file = file.split('_')
                    # print('******************************')
                    # print(file)
                    # print('******************************')
                    shape_paches_eachImage[key] = file[3]+'_'+file[5]
            # 遍历 files 列表中的每个文件名 file，如果 file 中包含当前键且包含 '_'+value+'_'，
            # 则将 file 以 _ 分隔成列表形式，取列表中第 4 个和第 5 个字段用 _ 连接起来，
            # 作为值存储到 shape_paches_eachImage 字典中，键为当前键。
        self.time_patches_eachImage = time_patches_eachImage
        self.num_patches_eachImage = num_patches_eachImage
        self.shape_paches_eachImage = shape_paches_eachImage
        # 最后将 time_patches_eachImage、num_patches_eachImage 和 shape_paches_eachImage
        # 分别赋值给类的属性 time_patches_eachImage、num_patches_eachImage 和 shape_paches_eachImage，
        # 并在最后输出了这三个字典的内容。
        # print(num_patches_eachImage)
        # print(shape_paches_eachImage)
        # print(time_patches_eachImage)

    def stack_image(self, save_path):
        for name in tqdm(self.landSet_name_test):        #对于测试集中每一张图片，用 tqdm 包装后遍历
            row, col = [int(i) for i in self.shape_paches_eachImage[name].split('_')]
            # 将该图片的行和列数从类属性 self.shape_paches_eachImage 中取出，
            # 该属性是一个字典，键为图片名，值为字符串，形如 '100_5'，表示该图片分成了 100 行，每行 5 个小图像。
            print(row,col)
            to_image = np.zeros((row * self.IMAGE_SIZE, col * self.IMAGE_SIZE))
            index = 1
            # 创建一个全零矩阵 to_image，大小为（行数 × 每个小图像的高度，列数 × 每个小图像的宽度），并初始化一个计数器 index 为 1。
            for y in range(1, row + 1): #对于每个小图像的行和列位置 y 和 x，从 1 遍历到对应的最大值。
                for x in range(1, col + 1):
                    # blue_patch_100_5_by_12_LC08_L1TP_064015_20160420_20170223_01_T1.TIF
                    image_file_name = 'nir_patch_{}_{}_by_{}_LC08_L1TP_{}_{}_01_T1.TIF'\
                        .format(index,y, x, name,self.time_patches_eachImage[name])
                    #根据计数器 index、行和列位置 y 和 x、图片名和该图片的时间属性，拼接出该小图像的文件名。
                    index += 1
                    #将计数器 index 加 1。
                    # for file in self.files:
                    #     if image_file_name in file:
                    #         image_file_name = file
                    #         break  #这里很重要
                    image_path = os.path.join(self.IMAGE_DIR, image_file_name)
                    from_image = Image.open(image_path)
                    mm = np.asarray(from_image)
                    ##构造该小图像文件的路径，并读取该文件为 Pillow 库的 Image 对象，再将其转换为 numpy 数组。
                    to_image[(y - 1) * self.IMAGE_SIZE:y * self.IMAGE_SIZE,
                    (x - 1) * self.IMAGE_SIZE:x * self.IMAGE_SIZE] = mm
                    #将该图片保存到磁盘上，文件名为图片名，并将灰度图像的 colormap 指定为灰度色。

            to_image = np.asarray(to_image)*255   #将拼接后图像的像素值扩大255倍
            to_image = Image.fromarray((to_image).astype(np.uint8))  #将 numpy 数组转换为 PIL 对象。
            # to_image.save('/media/estar/Data1/Lgy/FasterNet/results/{}.png'.format(name))
            # to_image = None
            #print(to_image.shape,to_image.max())
            path = save_path
            plt.imsave(path.format(name), to_image, cmap=plt.cm.gray)
            #将拼接后的图像保存为 png 文件。
            #print(path)

if __name__ == '__main__':
    save_path = 'connect_landsat8/{}.png'
    img_s = ImageStack('result_landsat8/')
    img_s.stack_image(save_path)