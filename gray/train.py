import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss #TCW20180913TCW
from models import DudeNet                                    # modify
from dataset import prepare_data, Dataset
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'    # modify

parser = argparse.ArgumentParser(description="DudeNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")  #  modify **********
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')

opt = parser.parse_args()
class sum_squared_error(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
def tv_norm(x,beta=4.0):
    img = x[0]
    dy = img -img
    dx = img -img
    dx[:,1:,:]=-img[:,:-1,:]+img[:,1:,:]
    dy[:,:,1:]= -img[:,:,:-1]+img[:,:,1:]
    return ((dx.pow(2)+dy.pow(2)).pow(beta/2.)).sum()

def main():
    # Load dataset
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DualNet(channels=1, num_of_layers=opt.num_of_layers)
    criterion = nn.MSELoss(size_average=False, reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load('logssigma25.0_2022-09-11-18-00-45yeganeh-without-cbam/model_60.pth'),
                          strict=True)
    criterion.to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    psnr_list = [] # it is used to save the psnr value of each epoch
    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > 30 and  epoch <=60:
            current_lr  =  opt.lr/10.     
        if epoch > 60  and epoch <=90:
            current_lr = opt.lr/100.
        if epoch >90 and epoch <=180:
            current_lr = opt.lr/1000.
        '''
        if epoch >120 and epoch  <=150:
            current_lr = opt.lr/10000.
        if epoch >150 and epoch <180:
            current_lr = opt.lr/100000.
        '''
        '''
         else:
            current_lr = opt.lr / 10.
        '''
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            img_train = data
            if opt.mode == 'S':
                 noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise

            img_train, imgn_train = Variable(img_train.to(device)), Variable(imgn_train.to(device))
            noise = Variable(noise.to(device))
            out_train = model(imgn_train)
            loss =  criterion(out_train, img_train) / (imgn_train.size()[0]*2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # model.eval()
            # results
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                torch.manual_seed(0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                img_val, imgn_val = Variable(img_val.to(device)), Variable(imgn_val.to(device),requires_grad=False)   # modify
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            psnr_val1 = str(psnr_val)
            psnr_list.append(psnr_val1)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
            # save model
            model_name = 'model'+ '_' + str(epoch+1) + '.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        filename = save_dir + 'psnr.txt'
        f = open(filename,'w')
        for line in psnr_list:
            f.write(line+'\n')
        f.close()
 
if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=41, stride=40, aug_times=4)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=41, stride=40, aug_times=4)
    main()
