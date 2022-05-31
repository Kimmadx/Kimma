import torch
import utils
import tqdm
import numpy as np
import torch
import time
import os
from cv2 import cv2
from imgaug import augmenters as iaa
from PIL import Image
import torchvision.transforms as transforms


def train(args, model, dataloader_train, dataloader_val, csv_path, datetime):
    str_P, str_R, str_F1 = [], [], []
    for i in range(args.num_classes):
        str_P.append(f'P{i}')
        str_R.append(f'R{i}')
        str_F1.append(f'F1_{i}')
    s = ('%15s;' * (6 + 3 * args.num_classes)) % (
    'epoch', 'loss for train', 'loss for val', 'PA', *str_P, *str_R, *str_F1, 'MPA', 'MIoU')  # 打印表头
    with open(os.path.join(args.base_dir, f'records/{args.data_name}/{args.model_name}.txt'), 'w') as f:
        f.write(s + '\n')

    mIoU_flag = 0.5
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-4)

    for epoch in range(args.num_epochs):
        if args.use_gpu:
            model = model.cuda()
        lr = utilsgai.poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        epoch += 1
        model.train()
        loss_record = []

        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)  # 查看进度
        tq.set_description(
            '%s(epoch:%d, lr:%f, dataset:%s)' % (args.model_name, epoch, lr, args.data_name))  # 查看每个进度条的描述，显示在进度条左侧

        for i, (data, label) in enumerate(dataloader_train):
            if args.use_gpu:
                data, label = data.cuda(), label.cuda()

            output = model(data)
            #output = output.logits
            loss = criterion(output, label)

            # output, output1, output2 = model(data)
            # loss = criterion(output, label) + criterion(output1, label) + criterion(output2, label)

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            loss_record.append(loss.item())

            tq.update(args.batch_size)  # 每次进度条更新的进度
            tq.set_postfix(loss='%.6f' % loss)  # 实时查看每条进度条的后缀，显示在进度条右侧
        tq.close()  # 关闭当前进度条

        loss_train_mean = np.mean(loss_record)
        print('loss for train:{:.6f}'.format(loss_train_mean))

        if args.save_each_model:
            torch.save(model.state_dict(),
                       os.path.join(args.base_dir, f'checkpoints/temporary_models_save/epoch_{epoch}.pth'))

        if epoch % args.validation_step == 0:
            mIoU = val(args, model, dataloader_val, csv_path, epoch, loss_train_mean, criterion)

        if mIoU > mIoU_flag:
            torch.save(model.state_dict(), os.path.join(args.base_dir,
                                                        'checkpoints/{}/{}/miou_{:.10f}_epoch_{}.pth'.format(
                                                            args.data_name, args.model_name, mIoU, epoch)))
            mIoU_flag = mIoU


def val(args, model, dataloader, csv_path, epoch, loss_train_mean, criterion):
    print('start val..........')
    predict_on_image(model, args, epoch, csv_path)
    start = time.time()
    with torch.no_grad():
        model.eval()
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        n_record = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            if args.use_gpu:
                data, label = data.cuda(), label.cuda()

            # get RGB predict image
            output = model(data)
            loss = criterion(output, label)

            loss_record.append(loss.item())
            predict = output.squeeze().permute(1, 2, 0)
            predict = torch.argmax(predict, dim=-1)
            predict = np.array(predict.cpu())
            # get RGB label image
            label = np.array(label.cpu())
            # compute per pixel accuracy
            hist += utilsgai.fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            N_record = utilsgai.coutpixel(label, predict, args.num_classes)
            for i in range(args.num_classes):
                for j in range(args.num_classes):
                    n_record[i][j] += N_record[i][j]
        loss_val_mean = np.mean(loss_record)
        sum_elements = np.sum(n_record)
        PA = np.sum([n_record[i][i] for i in range(args.num_classes)]) / sum_elements
        miou = np.mean(utilsgai.per_class_iu(hist))
        P = np.zeros(args.num_classes)
        for i in range(args.num_classes):
            P[i] += n_record[i][i] / n_record.sum(0)[i]
        R = np.zeros(args.num_classes)
        for i in range(args.num_classes):
            R[i] += n_record[i][i] / n_record.sum(1)[i]
        F1 = np.zeros(args.num_classes)
        for i in range(args.num_classes):
            F1[i] += 2 * P[i] * R[i] / (P[i] + R[i])
        MPA = np.mean(P)
        end = time.time()

        print('Time:{:.3f}s'.format(end - start))
        print('Loss:{:.5f}'.format(loss_val_mean))
        print('PA  :{:.5f}\nMPA :{:.5}\nMIoU:{:.5f}'.format(PA, MPA, miou))

        s = ('%15.5g;' * (6 + 3 * args.num_classes)) % (
        epoch, loss_train_mean, loss_val_mean, PA, *P, *R, *F1, MPA, miou)
        with open(os.path.join(args.base_dir, f'records/{args.data_name}/{args.model_name}.txt'), 'a') as f:
            f.write(s + '\n')
        return miou


def predict_on_image(model, args, epoch, csv_path):
    # pre-processing on image
    image = cv2.imread(os.path.join(args.base_dir, 'demo', f'{args.test_pic_name}.jpg'), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image).unsqueeze(0)
    # read csv label path
    label_info = utilsgai.get_label_info(csv_path)
    # predict
    model.eval()
    if args.use_gpu:
        image = image.cuda()
    predict = model(image)

    # predict, predict1, predict2 = model(image)

    w = predict.size()[-1]
    c = predict.size()[-3]
    predict = predict.resize(c, w, w).permute(1, 2, 0)
    predict = torch.argmax(predict, dim=-1)
    predict = utilsgai.colour_code_segmentation(np.array(predict.cpu()), label_info)
    predict = cv2.resize(np.uint8(predict), (args.crop_height, args.crop_width))
    save_path = os.path.join(args.base_dir, f'demo/{args.data_name}/{args.model_name}', 'epoch_%d.png' % (epoch))
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    print('==========Works==========')