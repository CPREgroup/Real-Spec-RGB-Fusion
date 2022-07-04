import torch
import pytorch_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_ssim(pred_y, true_y):

    ssim_value = pytorch_ssim.ssim(pred_y, true_y).item()
    return ssim_value


def psnr(pred_y, true_y):
    diff_square = (pred_y - true_y) ** 2
    mse = diff_square.mean()

    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    return 10 * torch.log10(PIXEL_MAX / mse)

def normalization(a):
    ran = a.max() - a.min()
    a = (a - a.min()) / ran
    return a, ran


