import shutil
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from skimage.transform import rotate

LABEL_TO_COLOR = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}


def make_image_dir(path_dir):
    path = Path(path_dir)
    # remove folder if it exists
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def mask2rgb(mask):
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)

    for i in np.unique(mask):
        rgb[mask == i] = LABEL_TO_COLOR[i]

    return rgb


def rgb2mask(rgb):
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k, v in LABEL_TO_COLOR.items():
        mask[np.all(rgb == v, axis=2)] = k

    return mask


def save_images(export_dir, data):
    save_dir_images = os.path.join(export_dir, 'images')
    save_dir_masks = os.path.join(export_dir, 'masks')

    make_image_dir(save_dir_images)
    make_image_dir(save_dir_masks)

    for i, (img, mask) in enumerate(data):
        save_image_path = os.path.join(save_dir_images, f'Img_{i}.png')
        save_mask_path = os.path.join(save_dir_masks, f'Mask_{i}.png')

        img = Image.fromarray(img)
        mask = Image.fromarray(mask2rgb(mask))

        img.save(save_image_path)
        mask.save(save_mask_path)


def threshold_otsu(x, *args, **kwargs) -> float:
    """Find the threshold value for a bimodal histogram using the Otsu method.

    If you have a distribution that is bimodal (AKA with two peaks, with a valley
    between them), then you can use this to find the location of that valley, that
    splits the distribution into two.

    From the SciKit Image threshold_otsu implementation:
    https://github.com/scikit-image/scikit-image/blob/70fa904eee9ef370c824427798302551df57afa1/skimage/filters/thresholding.py#L312
    """
    counts, bin_edges = np.histogram(x, *args, **kwargs)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def fft_downsize(x):
    scale = x.max()
    X = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
    N = X.shape[-1]
    z = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(X[:, :, N // 4:3 * N // 4, N // 4:3 * N // 4]), norm='forward'))
    return scale * z / z.max()


def fft_downsample(x):
    scale = x.max()
    X = np.fft.fftshift(np.fft.fft2(x, norm='forward'))
    N = X.shape[-1]
    Z = np.zeros_like(X)
    Z[:, N // 4:3 * N // 4, N // 4:3 * N // 4] = X[:, N // 4:3 * N // 4, N // 4:3 * N // 4]
    z = np.abs(np.fft.ifft2(np.fft.ifftshift(X), norm='forward'))
    return scale * z / z.max()


def fft_upsample(x):
    scale = x.max()
    X = np.fft.fftshift(np.fft.fft2(x, norm='forward'))
    N = X.shape[-1]
    X[:, N // 4:3 * N // 4, N // 4:3 * N // 4] = 0
    z = np.abs(np.fft.ifft2(np.fft.ifftshift(X), norm='forward'))
    return scale * z / z.max()


def rotate_over_axis0(arr, angle):
    N = arr.shape(0)
    out = np.empty_like(arr)
    for n in range(N):
        out[n] = rotate(arr[n], angle)


def calc_rmse(ref, img):
    return np.sqrt(np.mean(np.abs(ref - img) ** 2))


def calc_nmae(ref, img):
    return np.nanmean(np.abs(ref - img)/np.abs(ref))


def calc_psnr(ref, img):
    rmse = calc_rmse(ref, img)
    if rmse == 0.0:
        return 1000  # a high value because this has no meaning but I don't want to break the other codes with an np.Inf
    max_pixel = np.max(np.abs(ref))
    return 20 * np.log10(max_pixel / rmse)
