import os
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.filters import sobel


def calc_edge_similarity(c1_rolled, c2):
    c1_edges = sobel(c1_rolled)
    c1_flat = c1_edges.ravel().astype(float)
    c1_flat /= np.linalg.norm(c1_flat)

    c2_flat = c2.ravel().astype(float)
    c2_flat /= np.linalg.norm(c2_flat)

    dist = c1_flat @ c2_flat

    return dist


def align(c1, c2):
    h, w = c1.shape
    x_shift, y_shift = 0, 0

    # base case
    if not (w < 100 or h < 100):
        # downsampled images
        c1_small = resize(c1, (h // 2, w // 2), anti_aliasing=True)
        c2_small = resize(c2, (h // 2, w // 2), anti_aliasing=True)

        # recursive coarse alignment
        x_shift, y_shift, _ = align(c1_small, c2_small)

        # scale back up
        x_shift *= 2
        y_shift *= 2

    # local refinement around coarse estimate
    best_dist = -np.inf
    best_x, best_y = x_shift, y_shift
    best_img = c1

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            c2_edges = sobel(c2)
            shifted = np.roll(np.roll(c1, y_shift + dy, axis=0), x_shift + dx, axis=1)
            dist = calc_edge_similarity(shifted, c2_edges)
            if dist > best_dist:
                best_dist = dist
                best_x = x_shift + dx
                best_y = y_shift + dy
                best_img = shifted

    return best_x, best_y, best_img


def match_brightness(channel, target_mean, target_std):
    c_mean, c_std = np.mean(channel), np.std(channel)
    return ((channel - c_mean) / c_std) * (target_std + 1e-8) + target_mean


folder = "data"
# run alignment on all data images
for filename in os.listdir(folder):
    # set up files
    filepath = os.path.join(folder, filename)
    im = skio.imread(filepath)
    im = sk.img_as_float(im)
        
    # calculate 1/3 of total height
    height = np.floor(im.shape[0] / 3.0).astype(int) 
    width = im.shape[1]
    print(f"processing {filepath}, shape: " + str(im.shape))

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # cut edges off
    b = b[int(0.1*height):int(0.9*height), int(0.1*width):int(0.9*width)]
    r = r[int(0.1*height):int(0.9*height), int(0.1*width):int(0.9*width)]
    g = g[int(0.1*height):int(0.9*height), int(0.1*width):int(0.9*width)]

    # equalize brightness across color channels
    means = [np.mean(c) for c in [r, g, b]]
    stds = [np.std(c) for c in [r, g, b]]

    target_mean = np.mean(means)
    target_std = np.mean(stds)

    r = match_brightness(r, target_mean, target_std)
    g = match_brightness(g, target_mean, target_std)
    b = match_brightness(b, target_mean, target_std)

    # ensure still 0 -> 1 floats
    g = np.clip(g, 0, 1)
    r = np.clip(r, 0, 1)
    b = np.clip(b, 0, 1)

    # align g and r channels to b channel
    ag = align(g, b)[2]
    ar = align(r, b)[2]

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # crop borders for final image
    im_out = im_out[int(0.1*height):int(0.9*height), int(0.1*width):int(0.9*width), :]

    # save to output folder
    skio.imsave(f"sobel_out/{filepath.split("/")[1].split(".")[0]}_out.jpg", img_as_ubyte(im_out))