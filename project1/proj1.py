import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import img_as_ubyte
import os
from skimage.transform import resize
from skimage import io, img_as_ubyte

input_folder = "data"  
output_folder = "data_tif"  

# convert tifs to jpgs for website
"""
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif"):
        filepath = os.path.join(input_folder, filename)
        print(f"converting {filepath}...")

        im = io.imread(filepath)
        im_uint8 = img_as_ubyte(im)

        out_name = os.path.splitext(filename)[0] + ".jpg"
        out_path = os.path.join(output_folder, out_name)

        io.imsave(out_path, im_uint8)
"""

def calc_ncc(c1_rolled, c2):
    c1_flat = c1_rolled.ravel().astype(float)
    c1_flat /= np.linalg.norm(c1_flat)

    c2_flat = c2.ravel().astype(float)
    c2_flat /= np.linalg.norm(c2_flat)

    dist = c1_flat @ c2_flat

    return dist


def align(c1, c2):
    h, w = c1.shape
    x_shift, y_shift = 0, 0

    # base case
    if not (h < 120):
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
            shifted = np.roll(np.roll(c1, x_shift + dx, axis=0), y_shift + dy, axis=1)
            dist = calc_ncc(shifted, c2)
            if dist > best_dist:
                best_dist = dist
                best_x = x_shift + dx
                best_y = y_shift + dy
                best_img = shifted

    return best_x, best_y, best_img


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

    im_out_no_alignment = np.dstack([r, g, b])
    skio.imsave(f"original_out/{filepath.split("/")[1].split(".")[0]}_out.jpg", img_as_ubyte(im_out_no_alignment))

    # align g and r channels to b channel
    green_alignment = align(g, b)
    red_alignment = align(r, b)
    print("green filter shift: (" + str(green_alignment[0]) + ", " + str(green_alignment[1]) + ")")
    print("red filter shift: (" + str(red_alignment[0]) + ", " + str(red_alignment[1]) + ")\n")

    ag = green_alignment[2]
    ar = red_alignment[2]

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # crop borders for final image
    im_out = im_out[int(0.1*height):int(0.9*height), int(0.1*width):int(0.9*width), :]

    # save to output folder
    skio.imsave(f"out/{filepath.split("/")[1].split(".")[0]}_out.jpg", img_as_ubyte(im_out))