import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DudeNet
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure



parser = argparse.ArgumentParser(description="DudeNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='McMaster', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.
def vis(imgs):
    idx = torch.randint(0, 64, size=(1,))
    img = imgs[idx,...].abs().sum(1).squeeze(0)
    img = np.moveaxis(img.cpu().detach().numpy(), 0, -1)
    plt.imshow(img, interpolation='bilinear')


def main():
    # Build model
    print('Loading model ...\n')
    net = DudeNet(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_70.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    SSIM_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = torch.tensor(Img)
        Img = Img.permute(2,0,1)
        Img = Img.numpy()
        a1, a2, a3 = Img.shape
        Img = np.tile(Img,(3,1,1,1))
        Img = np.float32(normalize(Img))
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(12)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        #ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())#tcw
        #ISource = ISource.cuda()
        #INoisy = INoisy.cuda()
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource= ISource.to(device)
        INoisy = INoisy.to(device)
        with torch.no_grad():
            Out = torch.clamp(model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        SSIM = ssim(Out, ISource)
        psnr_test += psnr
        SSIM_test += SSIM
    #     *****************************
    # #     '''
    #     out = Out.permute(2,3,0,1)
    #     out = out[:,:,0,:]
    #     out = out.cpu().numpy()
    #     noisy_image = INoisy.permute(2,3,0,1)
    #     noisy_image = noisy_image[:,:,0,:]
    #     noisy_image = noisy_image.cpu().numpy()
    #     plt.subplot(1,2,1)
    #     noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    #     plt.imshow(noisy_image)
    #     plt.subplot(1,2,2)
    #     image = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    #     plt.imshow(image)
    #     # plt.show()
    #     # print()
    #     # '''

    psnr_test /= len(files_source)
    SSIM_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print("\nSSIM on test data %f" % SSIM_test)
def layers(imgs, model):
    layers = list(model.module.children())
    for i in range(22, 41):
        layer1 = nn.Sequential(*layers[22:i])
        out1 = layer1(imgs)
        out1 = out1.permute(2, 3, 1, 0)
        out1 = out1[:, :, :, 0]
        out1 = out1.cpu().detach().numpy()
        o1 = np.abs(out1[:, :, :]).sum(2)
        plt.title('layer' + str(i))
        plt.imshow(o1)
        # plt.show()

if __name__ == "__main__":
    main()
