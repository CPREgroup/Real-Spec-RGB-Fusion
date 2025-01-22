import os
import time
import sys


from argparse import ArgumentParser

parser = ArgumentParser(description='FullCNN-Net')

parser.add_argument('--flag',
                    # required=True,
                    default='fake_and_real_peppers_ms',
                    help="flag for log, or dataset img name for reminder")
parser.add_argument('--dataset', type=str, default="cave", help="datasetname")
parser.add_argument('--mis', type=str, default="unixy", help="reminder")
parser.add_argument('--gpuind', type=str, default="0", help="gpu for train")

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=1200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=6, help='phase number of ResCNN-Net')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--rgb_wei', type=float, default=1, help='ryb loss weight')

parser.add_argument('--model_dir', type=str, default='ablation/resblock', help='trained or pre-trained model directory')
# parser.add_argument('--log_dir', type=str, default='ablation/resblock', help='log directory')
parser.add_argument('--L', type=int, default=64, help='position encoding')
parser.add_argument('--eta', type=float, default=1.0, help='weight')
parser.add_argument('--ker_sz', type=int, default=8, help='kernel border size')
parser.add_argument('--imsz', type=int, default=512, help='rgb border size')
parser.add_argument('--hsi_slice_xy', type=str, default='0,0', help='check dataset_pre.py line 76-78 for explaination')

args = parser.parse_args()
log_dir = args.model_dir + '/trainlog'
args.model_dir = args.model_dir + '/trainmodel'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuind

import scipy.io
import torch
from torch.utils.data import DataLoader

from model_v2.dataset_pre import Dataset_pre, Dataset_pre_realdata
from model_v2.rollingnet import Fullcnn
from model_v2.tools import psnr, device
from skimage.metrics import peak_signal_noise_ratio as psnr2
import json


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num


model = Fullcnn(args)
# model = nn.DataParallel(model)
if device != "cpu":
    model = model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

datasetfoo = 'real' if 'real' in args.dataset else args.dataset
model_dir = f"./{args.model_dir}/VRCNN_{datasetfoo}_gauss{args.ker_sz}_{args.flag}"
log_file_name = f"./{log_dir}/VRCNN_{datasetfoo}_gauss{args.ker_sz}_{args.flag}.txt"


if not os.path.exists(f'./{log_dir}'):
    os.makedirs(f'./{log_dir}')
if not os.path.exists(model_dir):
    os.makedirs(f'{model_dir}/resimg')
    os.makedirs(f'{model_dir}/results')

dataset = Dataset_pre(args) if datasetfoo is not 'real' else Dataset_pre_realdata(args)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
print(len(dataset))

best_loss = 1e9

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./pretrained/pepper.pkl'))

for epoch_i in range(start_epoch + 1, end_epoch + 1):

    avg_loss = 0
    avg_psnr_cnn = 0
    avg_ssim = 0
    max_psnr = 0

    model.train()


    for j, (tmp_img, tmp_rgb, tmp_spec, tmp_pos, gt_kernel) in enumerate(train_loader):
        if j == 0:
            scipy.io.savemat(f'{model_dir}/results/input.mat', {
                'rgb': torch.squeeze(tmp_rgb).numpy(),
                'spec': torch.squeeze(tmp_spec).numpy()
            })

        m_loss = 0

        batch_img = tmp_img.float().to(device)  # its mask when dataset is 'real'
        batch_rgb = tmp_rgb.float().to(device)
        batch_spec = tmp_spec.float().to(device)
        batch_pos = tmp_pos.float().to(device)

        resspec_cnn, ryb_pred_cnn, spec_pred_cnn, phi, kernel = model(batch_rgb, batch_spec, batch_pos)


        # ryb_loss_cnn = 3 * (args.ker_sz**2) * torch.mean((ryb_pred_cnn - batch_rgb) ** 2)
        ryb_loss_cnn = criterion(ryb_pred_cnn, batch_rgb) * args.rgb_wei
        spec_loss_cnn = criterion(spec_pred_cnn, batch_spec)

        if args.ker_sz <= 4:
            tv_wei = 1.5
            lowrank_wei = 0.7
        else:
            # for real image
            tv_wei = 1.8
            lowrank_wei = 0.02

        tv_loss = 1 * 5e-6 * tv_wei * torch.sum(torch.abs(phi[1:, :] - phi[:-1, :])) # phi is SSF
        U, S, Vh = torch.linalg.svd(kernel, full_matrices=True)
        lr_loss = 1 * 1e-4 * lowrank_wei * torch.sum(S)
        tv_img_loss_cnn = 5e-7 * torch.sum(torch.abs(resspec_cnn[:, 1:, :, :] - resspec_cnn[:, :-1, :, :]))

        # loss_all = ryb_loss_cnn / ryb_loss_cnn.detach() + spec_loss_cnn / spec_loss_cnn.detach()
        loss_all = ryb_loss_cnn + spec_loss_cnn
        if datasetfoo == 'real':
            loss_all = loss_all + tv_loss # + lr_loss

        # s6 = time.time()
        # print('s6',s6-s5)

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if datasetfoo is not 'real':
            m_psnr_cnn = psnr(resspec_cnn, batch_img)
            avg_psnr_cnn += m_psnr_cnn

        avg_loss += loss_all.detach().item()

        outputdata_dic={
            'f': f'[{j}/{epoch_i}/{end_epoch}]'
            , 'Loss': loss_all.detach().item()
            , 'AVG Loss': f'{avg_loss / (j + 1):.7f}'
            , 'ryb CNN loss': f'{ryb_loss_cnn.detach().item():.7f}'
            , 'tv loss': f'{tv_loss.detach().item():.7f}'
            , 'lr loss': f'{lr_loss.detach().item():.7f}'
            , 'tv_img_CNNloss': f'{tv_img_loss_cnn.detach().item():.7f}'
            , 'time': time.strftime("%m-%d %H:%M", time.localtime())
        }
        if datasetfoo is not 'real':
            outputdata_dic['PSNR'] = m_psnr_cnn.detach().item()
            outputdata_dic['AVG CNN PSNR'] = f'{avg_psnr_cnn / (j + 1):.7f}'

        output_data = json.dumps(outputdata_dic)

        # s8 = time.time()
        # print('s8',s8-s7)

    # once
    if epoch_i % 50 == 0:
        mdic = {
            'phi': phi.detach().cpu().numpy(),
            'kernel': kernel.detach().cpu().numpy(),
            'gt_kernel': gt_kernel.detach().cpu().numpy()}
        scipy.io.savemat(f'{model_dir}/results/' + str(epoch_i) + '_' + str(j) + '.mat', mdic)

        # if m_psnr_cnn >= max_psnr:
        #     max_psnr = m_psnr_cnn
        #     print(f'max psnr is', m_psnr_cnn)




    with open(log_file_name, 'a') as f:
        f.write(output_data + '\n')
    #

    if avg_loss <= best_loss and epoch_i > 100:
        try:
            os.remove(last_modelname)
        except:
            print('no file')
        modelname=f"./{model_dir}/net_params_{epoch_i}.pkl"
        torch.save(model.state_dict(), modelname)  # save only the parameters
        last_modelname = modelname
        best_loss = avg_loss
        mdic = {'resimg_cnn': resspec_cnn.detach().cpu().numpy()}
        scipy.io.savemat(f'{model_dir}/resimg/resimg_' + str(epoch_i) + '_' + '.mat', mdic)
