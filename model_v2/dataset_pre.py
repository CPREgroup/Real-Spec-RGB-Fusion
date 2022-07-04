import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


from model_v2.PositionEncoding import positionencoding2D, positionalencoding2d
from model_v2.tools import normalization


class Dataset_pre(Dataset):
    def __init__(self, args):
        ksz = args.ker_sz

        # self.imgname = "img1/img1.mat"
        # self.imgname = f"../comparisons/datasets/{args.dataset}/{args.flag}.mat"
        self.imgname = f"./data/{args.flag}.mat"
        data = sio.loadmat(self.imgname)
        params = sio.loadmat('./ssf_ker_trained.mat')

        self.data, _ = normalization(np.transpose(data['hrhsi' if 'cave' in args.dataset else 'spec'], (2, 0, 1)))
        self.ssf = params['ssf']
        self.kernel = params[f'kernel{args.ker_sz}']


        self.rgb = np.matmul(self.ssf, self.data.reshape(31, 512*512)).reshape(3, 512, 512)

        # self.pos = positionencoding2D(512, 512, args.L, 'sin_cos')
        self.pos = positionalencoding2d(args.L * 4, 512, 512)

        args.imsz=512
        temp_data = np.transpose(self.data.reshape((31, 512 // ksz, ksz, 512 // ksz, ksz)), (0, 1, 3, 2, 4))
        self.spec = np.matmul(temp_data.reshape(31, 512 // ksz, 512 // ksz, ksz**2),self.kernel.reshape(ksz**2,1))
        self.spec = np.squeeze(self.spec, 3)

        count = 1

        print("dataset size: %d" %(count))

    def __getitem__(self, index):

        return self.data, self.rgb, \
               self.spec, self.pos, self.kernel #, np.transpose(PE[randx:randx+ksz, randy:randy+ksz, :],(2,0,1))

    def __len__(self):
        return len(self.data) #返回数据的总个数


class Dataset_pre_realdata(Dataset_pre):
    def __init__(self, args):
        # default setting is rgb is 3 times larger than hsi
        rgb_div_hsi = 6
        ksz = args.ker_sz


        self.imgname = f"../comparisons/datasets/{args.dataset}/"

        hsiname=glob.glob(f"{self.imgname}*hsi_small*")[0]
        rgbname=glob.glob(f"{self.imgname}*warped_small*")[0]

        hsi = sio.loadmat(hsiname)['sismall'][:,:,:31]
        rgb_warp = sio.loadmat(rgbname)
        rgb=rgb_warp['ryb']
        mask=rgb_warp['mask']

        # norm
        rgb=normalization(rgb)[0]
        hsi=normalization(hsi)[0]

        # split to ? x ?
        # shape is h w c
        hsiborder=128
        rgbborder=hsiborder*rgb_div_hsi
        h,w = list(map(int, args.hsi_slice_xy.split(',')))
        hsi = hsi[h:h+hsiborder, w:w+hsiborder, :]
        rgb = rgb[h*rgb_div_hsi:h*rgb_div_hsi+rgbborder, w*rgb_div_hsi:w*rgb_div_hsi+rgbborder, :]
        mask = mask[h*rgb_div_hsi:h*rgb_div_hsi+rgbborder, w*rgb_div_hsi:w*rgb_div_hsi+rgbborder]


        # viz
        # plt.imshow(rgb)
        # plt.show()
        # plt.imshow(hsi[:,:,18])
        # plt.show()


        self.data = mask
        self.kernel = np.array([0])


        self.rgb = rgb.transpose([2,0,1])

        self.spec = hsi.transpose([2,0,1])
        # downsample the spec
        # 512
        # self.spec = ttr.Resize(int(rgbborder / ksz),InterpolationMode.BICUBIC)(torch.from_numpy(self.spec))
        # 128 and 1536

        # downsample all pairs
        # 128 and 1536 -> 32 and 384
        # self.spec = ttr.Resize(32,InterpolationMode.BICUBIC)(torch.from_numpy(self.spec))
        # self.rgb = ttr.Resize(384,InterpolationMode.BICUBIC)(torch.from_numpy(self.rgb))

        c,h,w=self.rgb.shape
        self.pos = positionalencoding2d(args.L * 4, h, w)
        args.imsz=h


        count = 1

        print("dataset size: %d" %(count))

    def __len__(self):
        return 31

