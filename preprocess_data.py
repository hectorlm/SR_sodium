import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib import pyplot as plt
plt.ion()
import nibabel as nib
from skimage.morphology import remove_small_holes, remove_small_objects
from glob import glob
from skimage.transform import resize
from lib.utils import rotate as rot
from lib.utils import fft_downsize, fft_downsample

morph = lambda x, axis: remove_small_holes(remove_small_objects(x))
# Loop through the .nii files in data
datadir = "./data/"
files = ["rNa_lr.nii", "PD.nii", "T1.nii", "T2.nii", "rNa_hr.nii"]
train_it = 0
test_it = 0
for i in range(1, 8):
    print(f'Loading masks for file: {i}')
    mask = nib.load(f'./data/Masks/Data{i}/c1PD.nii').get_fdata()
    mask += nib.load(f'./data/Masks/Data{i}/c2PD.nii').get_fdata()
    mask += nib.load(f'./data/Masks/Data{i}/c3PD.nii').get_fdata()
    mask = mask > 0.0
    mask = np.apply_over_axes(morph, mask, [1, 2])
    print(f'Working on file: {i}')
    img = nib.load(f'{datadir}{i}{files[0]}').get_fdata()
    img[np.isnan(img)] = 0.0
    # img = img * mask
    img /= img.max()
    mmag = np.mean(img, (1, 2))
    idx = mmag > (np.max(mmag) * 0.4)

    # imglr = resize(img[idx], [sum(idx), 80, 80], anti_aliasing=True)
    # imglr = fft_downsample(img)

    bimg = np.zeros((4, np.sum(idx), 160, 160))
    bimg[0] = img[idx]
    for jj in range(1, 4):
        img = nib.load(f'{datadir}{i}{files[jj]}').get_fdata()
        img[np.isnan(img)] = 0.0
        # img = img * mask
        img /= img.max()
        bimg[jj] = img[idx]

    img2 = nib.load(f'{datadir}{i}{files[-1]}').get_fdata()
    img2[np.isnan(img2)] = 0.0
    img2 = img2 * mask
    img2 /= img2.max()
    imghr = img2[idx]

    for k in range(sum(idx)):
        if i <= 6:
            ang = np.arange(-180, 180, 20)
            for n in range(10):
                np.savez(f'{datadir}train/{train_it}', lr=rot(bimg[:, k], ang[n]), hr=rot(imghr[k], ang[n]))
                train_it += 1
        else:
            np.savez(f'{datadir}test/{test_it}', lr=bimg[:, k], hr=imghr[k])
            test_it += 1

# files = glob('./data/proton_only/*.nii')
# for i, file in enumerate(files):
#     print(f'Working on file: {file}')
#     img = nib.load(file).get_fdata()
#     img /= img.max()
#     mmag = np.mean(img, (1, 2))
#     idx = mmag > (np.max(mmag) * 0.4)
#     slices = img.shape[0]
#     # nimg = np.zeros((img.shape[0], img.shape[1], 160), dtype=img.dtype)
#     # nimg[:, :, 0:155] = img
#     imghr = resize(img, [slices, 160, 160], anti_aliasing=True)
#     imglr = fft_downsample(imghr)
#     # imglr = resize(img, [slices, 80, 80], anti_aliasing=True)
#     imghr = imghr[idx]
#     imglr = imglr[idx]
#     for k in range(sum(idx)):
#         np.savez(f'{datadir}train/{train_it}', lr=imglr[k], hr=imghr[k])
#         train_it += 1

