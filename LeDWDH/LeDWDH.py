from datetime import datetime
import math

import albumentations as A
import fastmri
from albumentations.pytorch import ToTensorV2
from numpy import mean
from torch.utils.tensorboard import SummaryWriter
import torch, gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import time

import torchvision.transforms as transforms
import random
import torch.backends.cudnn as cudnn

from model.YNet import YNet

from logs.logger import LOG, get_timestamp

gc.collect()
torch.cuda.empty_cache()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2

TRAIN_HLM_DIR = "./data/train/wrapped_phase_638.65/"
TRAIN_AMP_DIR = "./data/train/wrapped_phase_665.00/"
TRAIN_LABEL_DIR = "./data/train/white_light/"

TEST_HLM_DIR = "./data/test/wrapped_phase_638.65/"
TEST_AMP_DIR = "./data/test/wrapped_phase_665.00/"
TEST_LABEL_DIR = "./data/test/white_light/"

CKPT_DIR = "./ckpt/"
RESULTS_DIR = "./results/"

learning_rate = 1e-8
schedule = [500, 700]
COS = False
filename = 'tensorholo.pt'

BATCH_SIZE = 1
EPOCH = 801

TEST_CKPT_NAME = 'tensorholo.pt'

def tv_loss(x, beta=0.5):

    x = x.cuda()
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(dh[:, :, :-1] + dw[:, :, :, :-1])

def get_file(path):
    files = os.listdir(path)
    files.sort(key=lambda x:int(x[:-4]))  # 排序
    list = []
    for file in files:
        if not os.path.isdir(path + file):  # 判断该文件是否是一个文件夹
            f_name = str(file)
            filename = path + f_name
            list.append(filename)
    return list

def hlm_file_list(is_test):
    if is_test==True:
        PATH = TEST_HLM_DIR

    else:
        PATH = TRAIN_HLM_DIR
    hlm_list=get_file(PATH)
    return hlm_list

def amp_file_list(is_test):
    if is_test==True:
        PATH = TEST_AMP_DIR
    else:
        PATH = TRAIN_AMP_DIR
    amp_list=get_file(PATH)
    return amp_list

def label_file_list(is_test):
    if is_test==True:
        PATH = TEST_LABEL_DIR
    else:
        PATH = TRAIN_LABEL_DIR
    label_list=get_file(PATH)
    return label_list

class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    def __call__(self, img):
        return self.data_transform(img)

class TensorHoloDataset(torch.utils.data.Dataset):
    def __init__(self, hlm_list, amp_list, label_list, transform):
        self.hlm_list = hlm_list
        self.amp_list = amp_list
        self.label_list = label_list
        self.transform = transform
        #self.transform=None

    def __len__(self):
        return len(self.hlm_list)
    
    def __getitem__(self, index):
        hlm_path = self.hlm_list[index]
        hlm = cv2.imread(hlm_path, 2)
        amp_path = self.amp_list[index]
        amp = cv2.imread(amp_path,2)
        label_path = self.label_list[index]
        label = cv2.imread(label_path, 2)

        hlm = np.expand_dims(hlm, axis=2)
        amp = np.expand_dims(amp, axis=2)
        label = np.expand_dims(label, axis=2)
        cat = np.append(hlm, amp, axis=2)
        cat = np.append(cat, label, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=cat)
            cat = transformed["image"]
            hlm = cat[0, :, :]
            hlm = hlm[None, :, :]
            amp = cat[1, :, :]
            amp = amp[None, :, :]
            label = cat[2, :, :]
            label = label[None, :, :]
        return hlm, amp, label

def rescale(cgh):
    min_cgh = np.min(cgh)
    max_cgh = np.max(cgh)

    cgh = (cgh - min_cgh) / (max_cgh - min_cgh)
    return cgh


device = torch.device("cuda")
print(device)

net = YNet(in_channels=1, features=64).to(device)
net = net.to(device)

# criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

# criterion1.to(device)
criterion2.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

hlm_list = hlm_file_list(is_test=False)
amp_list = amp_file_list(is_test=False)
label_list = label_file_list(is_test=False)

transfroms_A = A.Compose(
    [
        ToTensorV2(),
    ]
)
def train(epoch):
    train_dataset = TensorHoloDataset(hlm_list=hlm_list, amp_list=amp_list, label_list=label_list,  transform=transfroms_A)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if epoch % 100 == 0:
        print('\n[ Train Epoch: %d ]'%epoch)

    train_loss_list = []
    loss_tv_list = []
    loss1_list = []
    loss_list = []

    for batch_idx, (train_hlm,train_amp,train_label) in enumerate(train_loader):
        train_hlm = train_hlm.to(device)
        train_hlm = train_hlm - torch.min(train_hlm)
        train_hlm = train_hlm

        train_amp = train_amp.to(device)
        train_amp = train_amp - torch.min(train_amp)
        train_amp = train_amp

        input1 = torch.cat((train_hlm, train_amp), dim=1)

        train_label = train_label.to(device)
        train_label = train_label - torch.min(train_label)

        # input1 = torch.cat((train_hlm, train_amp), dim=1)
        # output = net(input1)

        output = net(train_hlm, train_amp)  # YNet input

        #'''
        loss_amp = criterion2(output, train_label)  # l1_loss

        err = output - train_label
        rmse = torch.std(err)
        loss_2 = 1e-7 * tv_loss(output)  # TV
        loss = loss_amp + loss_2

        if epoch % 100 == 0:
            if batch_idx >= 0 and batch_idx % 1 == 0:
                os.makedirs('./results/epoch_{}/output'.format(epoch), exist_ok=True)
                RESULTS_DIR = './results/'
                RESULTS_DIR_AMP = RESULTS_DIR + 'epoch_{}/output/'.format(epoch)

                output1 = output.detach().cpu().numpy()
                output1 = np.transpose(output1[0, :, :, :], [1, 2, 0])

                cv2.imwrite(os.path.join(RESULTS_DIR_AMP, '%d.tif' % (batch_idx + 1)), output1)

                os.makedirs('./results/epoch_{}/train_label'.format(epoch),
                            exist_ok=True)
                RESULTS_label_DIR = './results/'
                RESULTS_label_DIR_AMP = RESULTS_label_DIR + 'epoch_{}/train_label/'.format(epoch)
                train_label1 = train_label.detach().cpu().numpy()
                train_label1 = np.transpose(train_label1[0, :, :, :], [1, 2, 0])
                cv2.imwrite(os.path.join(RESULTS_label_DIR_AMP, '%d.tif' % (batch_idx + 1)), train_label1)

                os.makedirs('./results/epoch_{}/train_input'.format(epoch),
                            exist_ok=True)
                RESULTS_input1_DIR = './results/'
                RESULTS_input1_DIR_AMP = RESULTS_input1_DIR + 'epoch_{}/train_input1/'.format(epoch)
                input11 = input1.detach().cpu().numpy()
                input111 = input11[0, 0, :, :]
                input111 = input111[None, :, :]
                input1111 = np.transpose(input111, [1, 2, 0])
                cv2.imwrite(os.path.join(RESULTS_input1_DIR_AMP, '%d.tif' % (batch_idx + 1)), input1111)
                RESULTS_input2_DIR = './results/'
                RESULTS_input2_DIR_AMP = RESULTS_input2_DIR + 'epoch_{}/train_input2/'.format(epoch)
                # print(input11.shape)

                # input222 = input11[0, 1, :, :]  # psww1 psww2 as input
                # input222 = input222[None, :, :] # psww1 psww2 as input

                input222 = input11[0, 1, :, :]  # sub_H11 as input
                input222 = input222[None, :, :]

                input2222 = np.transpose(input222, [1, 2, 0])
                cv2.imwrite(os.path.join(RESULTS_input2_DIR_AMP, '%d.tif' % (batch_idx + 1)), input2222)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        loss1_list.append(loss_amp.item())
        loss_tv_list.append(loss_2.item())
        train_loss_list.append(rmse.item())

    if epoch % 100 == 0:
        print(datetime.now().strftime('%y-%m-%d-%H.%M.%S'))
        print('loss = :', mean(loss_list))  # 800 -> len(train data)
        print('l1_loss = :', mean(loss1_list))  # 800 -> len(train data)
        print('loss_tv =:', mean(loss_tv_list))
        print('RMSE:', mean(train_loss_list))

    if epoch % 100 == 0:
        state={
            # "epoch": epoch+1,
            "model_state_dict":net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }
        torch.save(state, './ckpt/' + filename)
        print('Model Saved!')

    loss_train_closure = [mean(loss_list), mean(loss1_list), mean(loss_tv_list), mean(train_loss_list)]
    return loss_train_closure

def test(epoch):
    hlm_list = hlm_file_list(is_test=True)
    amp_list = amp_file_list(is_test=True)
    label_list = label_file_list(is_test=True)
    test_dataset = TensorHoloDataset(hlm_list=hlm_list, amp_list=amp_list, label_list=label_list, transform=A.Compose([ToTensorV2()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    time0=time.time()
    
    # net.eval()
    test_loss = 0
    if epoch % 100 == 0:
        if not os.path.isdir('./results/epoch_{}/val_output'.format(epoch)):
            os.makedirs('./results/epoch_{}/val_output'.format(epoch), exist_ok=True)
        RESULTS_DIR_AMP = './results/epoch_{}/val_output/'.format(epoch)
        if not os.path.isdir('./results/epoch_{}/val_label'.format(epoch)):
            os.makedirs('./results/epoch_{}/val_label'.format(epoch), exist_ok=True)
        RESULTS_DIR_label_AMP = './results/epoch_{}/val_label/'.format(epoch)
        if not os.path.isdir('./results/epoch_{}/val_input'.format(epoch)):
            os.makedirs('./results/epoch_{}/val_input'.format(epoch), exist_ok=True)
        RESULTS_DIR_input1_AMP = './results/epoch_{}/val_input1/'.format(epoch)
        if not os.path.isdir('./results/epoch_{}/val_input1'.format(epoch)):
            os.makedirs('./results/epoch_{}/val_input1'.format(epoch), exist_ok=True)
        RESULTS_DIR_input2_AMP = './results/epoch_{}/val_input2/'.format(epoch)
        if not os.path.isdir('./results/epoch_{}/val_input2'.format(epoch)):
            os.makedirs('./results/epoch_{}/val_input2'.format(epoch), exist_ok=True)

    test_loss_list, test_loss1_list, loss_tv_list, test_rmse_list = [], [], [], []
    for batch_idx, (test_hlm, test_amp, test_label) in enumerate(test_loader):
            test_hlm = test_hlm.to(device)
            test_hlm = test_hlm - torch.min(test_hlm)
            test_hlm = test_hlm

            test_amp = test_amp.to(device)
            test_amp = test_amp - torch.min(test_amp)
            test_amp = test_amp

            test_label = test_label.to(device)
            test_label = test_label - torch.min(test_label)

            input_test = torch.cat((test_hlm, test_amp), dim=1)

            # output_test = net(input_test)
            output_test = net(test_hlm, test_amp)  # YNet input

            loss_amp = criterion2(test_label, output_test)

            loss_2 = 1e-7 * tv_loss(output_test)  # TV
            err_test = output_test - test_label
            rmse_test = torch.std(err_test)

            loss = loss_amp + loss_2

            test_loss_list.append(loss.item())
            test_loss1_list.append(loss_amp.item())
            loss_tv_list.append(loss_2.item())
            test_rmse_list.append(rmse_test.item())

            if epoch % 100 == 0:
                if batch_idx >= 0 and batch_idx % 1 == 0:
                    input_test1 = input_test.detach().cpu().numpy()
                    output_test = output_test.detach().cpu().numpy()
                    test_label1 = test_label.detach().cpu().numpy()

                    input_test11 = input_test1[0, 0, :, :]
                    input_test111 = input_test11[None, :, :]

                    input_test1111 = np.transpose(input_test111, [1, 2, 0])

                    input_test22 = input_test1[0, 1, :, :] # sub_H11 as input
                    input_test222 = input_test22[None, :, :]    # sub_H11 as input
                    # print(input111.shape)
                    input_test2222 = np.transpose(input_test222, [1, 2, 0])

                    output_amp=np.transpose(output_test[0,:,:,:], [1,2,0])
                    test_label1=np.transpose(test_label1[0,:,:,:], [1,2,0])

                    cv2.imwrite(os.path.join(RESULTS_DIR_input1_AMP, '%d.tif' % (batch_idx + 1)), input_test1111)
                    cv2.imwrite(os.path.join(RESULTS_DIR_input2_AMP, '%d.tif' % (batch_idx + 1)), input_test2222)
                    cv2.imwrite(os.path.join(RESULTS_DIR_AMP, '%d.tif' % (batch_idx + 1)), output_amp)
                    cv2.imwrite(os.path.join(RESULTS_DIR_label_AMP, '%d.tif' % (batch_idx + 1)), test_label1)


    dt=time.time()-time0
    if epoch % 100 == 0:
        print('\n[ Test Epoch: %d ]' % epoch)
        print('test_LOSS: ', mean(test_loss_list))  # 200 -> len(test data)
        print('test_L1_LOSS: ', mean(test_loss1_list))  # 200 -> len(test data)
        print('test_LOSS_tv: ', mean(loss_tv_list))  # 200 -> len(test data)
        print('test_RMSE: ', mean(test_rmse_list))  # 200 -> len(test data)
        print('iter time {:0.5f}'.format(dt))
    loss_test_closure = [mean(test_loss_list), mean(test_loss1_list), mean(loss_tv_list), mean(test_rmse_list)]
    return loss_test_closure


def adjust_learning_rate(optimizer, epoch, lr, cos, epochs, schedule):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch % 100 == 0:
        print('lr={}'.format(lr))


if __name__=='__main__':
    print(datetime.now().strftime('%y-%m-%d-%H.%M.%S'))

    if os.path.isfile(CKPT_DIR + TEST_CKPT_NAME):
        print("Loading Checkpoint")
        checkpoint = torch.load(CKPT_DIR + TEST_CKPT_NAME)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = 0

        global_step = checkpoint["global_step"]
    else:
        print("No checkpoint, start form the zero")
        start_epoch = 0
        global_step = 0

    if start_epoch==EPOCH:
        test(start_epoch)

    save_path = './logs/{}'.format(get_timestamp())
    os.makedirs(save_path, exist_ok=True)
    log = LOG(save_path, filename='training_loss',
              field_name=['epoch', 'train_loss', 'train_loss1', 'train_loss2(TV)', 'train_mse', 'test_loss', 'test_loss1', 'test_loss2(TV)', 'test_mse'])
    for epoch in range(start_epoch, EPOCH):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate, cos=COS, epochs=EPOCH, schedule=schedule)
        # start = time.time()

        loss_train_avg = train(epoch)
        # end = time.time()
        # print('start:', start)
        # print('end:', end)
        # print('time:', (end - start))
        if epoch % 1 == 0:
            loss_test_avg = test(epoch)
        log.record(epoch + 1, *loss_train_avg, *loss_test_avg)
    log.close()
    