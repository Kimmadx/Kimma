import torch
import pandas as pd
import numpy as np
from PIL import Image
import torch.utils.data as Data
import random
import os
from glob import glob
import cv2
from torchvision import transforms
from imgaug import augmenters as iaa


class SG(Data.Dataset):
    def __init__(self, img_path, label_path, csv_path):
        super().__init__()
        self.img_list = glob(os.path.join(img_path,'*.jpg'))
        self.label_list = glob(os.path.join(label_path,'*.png'))
        self.label_info = get_label_info(csv_path)
        self.colormap = [self.label_info[key] for key in self.label_info]

        try:
            assert (len(self.img_list))==(len(self.label_list))
        except AssertionError as AE:
            print('Samples and labels are not match!')
        random.seed(32)
        random.shuffle(self.img_list)
        random.seed(32)
        random.shuffle(self.label_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.img_transforms(img, label, self.colormap)
        return img, label.long()

    def __len__(self):
        return len(self.img_list)

    def img_transforms(self, data, label, colormap):
        data = transforms.ToTensor()(data)
        label = torch.from_numpy(image2label(label, colormap))
        return data, label





def get_label_info(csv_path='D:/semsg(jiqixuexi)/datasets/land_seg4.0/class_dict.csv'):
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name=row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


def image2label(image, colormap):
    cm2lbl = np.zeros(256**3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0]*256*256 + cm[1]*256 + cm[2])] = i
    image = np.array(image,dtype='int64')
    np.set_printoptions(threshold=np.inf)
    ix = (image[:,:,0]*256*256 + image[:,:,1] * 256 + image[:,:,2])
    image2 = cm2lbl[ix]
    return image2

def predict_on_image(model, height,width, csv_path,read_path,save_path):
    # pre-processing on image
    # image = cv2.imread("demo/ceshi.png", -1)
    image = cv2.imread(read_path, -1)  # 读进来直接是BGR(不
    # 是我们最常见的RGB格式，颜色肯定有区别。) 格式数据格式在 0~255,
    # flag = -1,   8位深度，原通道
    # flag = 0，   8位深度，1通道
    # flag = 1，   8位深度，3通道
    # flag = 2，   原深度， 1通道
    # flag = 3，   原深度， 3通道
    # flag = 4，   8位深度，3通道

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    resize = iaa.Scale({'height': height, 'width': width})  # 将图像缩放到固定大小
    # 将增强序列应用于图像和关键点。 为了确保以相同的方式增加两种数据类型，我们首先将增强序列切换到确定模式。 没有它，我们会获得图像和关键点间不匹配旋转。 请注意，每转换一批切换一次确定模式。
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')  # Opencv转PIL.Image
    image = transforms.ToTensor()(image).unsqueeze(0)  # 加一个batch_size 的维度，这样才能放进模型里去跑
    # read csv label path
    label_info = get_label_info(csv_path)
    # predict
    model.eval()
########################单loss输出###################################
    predict = model(image.cuda())[0]
########################多loss输出################################
    # predict = model(image.cuda())[0]
##################################################################
    #with torch.no_grad():
        #image1 = cv2.imread("demo/ceshi.png", -1)
        #predict = model(image.cuda())
        #predict=predict.cpu().numpy()
        #predict=predict[0,1,:,:]


        #pmin=np.min(predict)
        #pmax=np.max(predict)
        #predict=((predict-pmin)/(pmax-pmin+0.000001))*225
        #predict=predict.astype(np.uint8)
        #predict=cv2.applyColorMap(predict,cv2.COLORMAP_JET)
        #predict=predict[:,:,::-1]
        #predict = image1+predict*0.3
        #plt.imshow(predict, cmap='gray')
        #save_path = 'demo/epoch_%d.png' % (epoch)
        #cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    w =predict.size()[-1]
    h =predict.size()[-2]
    c =predict.size()[-3]
    predict = predict.resize(c,h,w)
    predict = reverse_one_hot(predict)  # (h,w)一张图上面每一个像素点对应分类好的数字
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 效果看123.py
    predict = cv2.resize(np.uint8(predict), (height, width))  # 用cv2读图的时候，已经转换成了 (h,w,c)格式了
    # save_path = 'demo/epoch_%d.png' % (epoch+1)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))  # 指定路径存储图片






def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=600, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    else:
        lr = init_lr * (1 - iter / max_iter) ** power
        optimizer.param_groups[0]['lr'] = lr
        return lr

def reverse_one_hot(image):
    image = image.permute(1, 2, 0)   # [2, 512, 512] ==> [512, 512, 2]
    x = torch.argmax(image, dim=-1)  # [512, 512, 2] ==> [512, 512]
    return x



def colour_code_segmentation(image, label_values):
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]
	return x


def fast_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
	epsilon = 1e-5
	return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def coutpixel(y_true, y_predict, classes):
	N = np.zeros((classes, classes))
	for i in range(classes):
		for j in range(classes):
			N[i][j] += np.sum((y_true == i) & (y_predict == j))
	return N


if __name__ == '__main__':
    print('==========Utils==========')