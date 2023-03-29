import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib import pyplot as plt
plt.ion()
# plt.close('all')
import nibabel as nib
import torch
import torch.nn as nn
from model import UNet, ResNet
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation
from lib.utils import calc_rmse, calc_psnr, calc_nmae
from lib.utils import rotate as rot

#%%
morph = lambda x, axis: remove_small_holes(remove_small_objects(x))
# Loop through the .nii files in data
datadir = "./data/"

files = ["rNa_lr.nii", "PD.nii", "T1.nii", "T2.nii", "rNa_hr.nii"]

device = ("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet(n_channels=4, reps=2)
net.load_state_dict(torch.load('./checkpoints/Res2UNET_SR_L1001L2_TEST-lr0.05Ba8.pth'))
# net = UNet(n_channels=4)
# net.load_state_dict(torch.load('./checkpoints/ResUNET_SR_L1_N_masked-lr0.1Ba32.pth'))
net.to(device)
net.eval()

i = 4 # 1,2,3,4
for i in range(1, 8):
    print(f'Loading masks for file: {i}')
    mask = nib.load(f'./data/Masks/Data{i}/c1PD.nii').get_fdata()
    mask += nib.load(f'./data/Masks/Data{i}/c2PD.nii').get_fdata()
    mask += nib.load(f'./data/Masks/Data{i}/c3PD.nii').get_fdata()
    mask = mask > 0.0
    mask = np.apply_over_axes(morph, mask, [1, 2])
    mask = binary_dilation(mask)
    print(f'Working on file: {i}')
    img = nib.load(f'{datadir}{i}{files[0]}').get_fdata()
    img[np.isnan(img)] = 0.0
    img = img*mask
    img /= img.max()
    mmag = np.mean(img, (1, 2))
    idx = mmag > (np.max(mmag) * 0.4)

    bimg = np.zeros((np.sum(idx), 4, 160, 160))
    bimg[:, 0] = img[idx]
    for jj in range(0, 4):
        img = nib.load(f'{datadir}{i}{files[jj]}').get_fdata()
        img[np.isnan(img)] = 0.0
        img = img * mask
        img /= img.max()
        bimg[:, jj] = img[idx]

    # from scipy.signal import convolve2d
    # tumor = np.zeros(bimg.shape[-2:])
    # tumor[62:66, 71:75] = 0.15
    # gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
    #                                   [2., 4., 2.],
    #                                   [1., 2., 1.]])
    # tumor = convolve2d(tumor, gaussian, 'same')
    # bimg[:, 0] += tumor

    # bimg[:, 0] = 0

    protimg = bimg[8:14]
    imglr = bimg[8:14, 0]

    img2 = nib.load(f'{datadir}{i}{files[-1]}').get_fdata()
    img2[np.isnan(img2)] = 0.0
    img2 = img2*mask
    img2 /= img2.max()
    imghr = img2[idx, :, :]
    imghr = np.maximum(0, imghr)
    imghr = imghr[8:14]

    mask2 = mask[idx]

    timglr = torch.as_tensor(bimg[8:14].astype(np.float32)).to(device)
    imgsr = net(timglr)
    imgsr = imgsr.cpu().detach().numpy().squeeze()*mask2[8:14]

    imgsr = imgsr/imgsr.max()

    from piqa import SSIM
    ssim = SSIM(n_channels=1)
    x = torch.as_tensor(np.expand_dims(imghr.astype(np.float32), 1))
    y = torch.as_tensor(np.expand_dims(imglr.astype(np.float32), 1))
    z = torch.as_tensor(np.expand_dims(imgsr.astype(np.float32), 1))

    fig, axs = plt.subplots(3, 7)
    pt = [0, imglr.shape[0]//2-1, imglr.shape[0]-1]
    for j in range(0, 3):#imgsr.shape[0]):
        axs[j, 0].imshow(rot(imglr[pt[j], ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 0].axis('off')
        axs[j, 1].imshow(rot(imghr[pt[j], ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 1].axis('off')
        axs[j, 2].imshow(rot(imgsr[pt[j], ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 2].axis('off')
        axs[j, 3].imshow(rot(np.abs(imghr-imgsr)[pt[j], ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 3].axis('off')
        axs[j, 4].imshow(rot(protimg[pt[j], 1, ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 4].axis('off')
        axs[j, 5].imshow(rot(protimg[pt[j], 2, ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 5].axis('off')
        axs[j, 6].imshow(rot(protimg[pt[j], 3, ::1, ::1], 90), cmap='gray', vmin=0, vmax=1, interpolation='spline16')
        axs[j, 6].axis('off')
        if j == 0:
            axs[j, 0].set_title(f'Bicubic Interpolation')
            axs[j, 1].set_title('Reference')
            axs[j, 2].set_title(f'Super Resolved')
            axs[j, 3].set_title(f'Error')
            axs[j, 4].set_title('PD')
            axs[j, 5].set_title(f'T1')
            axs[j, 6].set_title(f'T2')
        fig.set_tight_layout('tight')

    print(f'Upsampled image - SSIM = {ssim(x,y):.3f}')
    print(f'Upsampled image - RMSE = {calc_rmse(imghr,imglr):.1e}')
    print(f'Upsampled image - NMAE = {calc_nmae(imghr,imglr):.1e}')
    print(f'Upsampled image - PSNR = {calc_psnr(imghr,imglr):.1f}')
    print(f'Super resolved image - SSIM = {ssim(x,z):.3f}')
    print(f'Super resolved image - RMSE = {calc_rmse(imghr,imgsr):.1e}')
    print(f'Super resolved image - NMAE = {calc_nmae(imghr,imgsr):.1e}')
    print(f'Super resolved image - PSNR = {calc_psnr(imghr,imgsr):.1f}')
