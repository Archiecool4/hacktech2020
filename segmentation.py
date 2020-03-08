import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

import numpy as np

from skimage import io, color, filters, feature, morphology, exposure
from skimage.transform import downscale_local_mean

from scipy import ndimage as ndi

import os

import pickle

from crop import crop_image


def get_cropped(name, new=False):
    # If a new subject, call the Cloud Vision API
    if new:
        crop_image(name)

    polygon = None

    # Load bounding box
    with open(name + '.p', 'rb') as f:
        polygon = pickle.load(f)

    polygon = np.array([[p.x, p.y] for p in polygon])

    # Load image and darken the image
    image = io.imread(name + '.jpg')
    image = exposure.adjust_gamma(image, gamma=1.5, gain=1)

    x = image.shape[1]
    y = image.shape[0]

    # Draw a rectangle to crop the image
    rect = mpatches.Rectangle((polygon[0][0] * x, polygon[0][1] * y), 
        polygon[1][0] * x - polygon[0][0] * x, polygon[2][1] * y - 
        polygon[0][1] * y, linewidth=1, edgecolor='r', facecolor='none')

    bbox = rect.get_bbox()
    cropped = image[int(bbox.y0):int(bbox.y1), int(bbox.x0):int(bbox.x1)]

    return cropped


def fill_in_edges(im):
    # Loop through rows and columns and fill in missing edges iff more than 
    # half of that column is filled with 1s (edge present)
    for i in range(len(im[0])):
        col = im[:, i]
        if np.sum(col) > (0.5 * len(im)):
            first = np.argmax(col == 1)
            last = len(col) - 1 - np.argmax(col[::-1] == 1)
            im[:, i][first:last] = np.ones(last - first)
    for i in range(len(im)):
        row = im[i, :]
        if np.sum(row) > (0.5 * len(im[0])):
            first = np.argmax(col == 1)
            last = len(row) - 1 - np.argmax(row[::-1] == 1)
            im[i, :][first:last] = np.ones(last - first)


def process_image(name, new=False):
    cropped = get_cropped(name, new)

    # Grayscale
    gray = color.rgb2gray(cropped)

    # Use a factor of 20 to downscale the image
    image_downscaled = downscale_local_mean(gray, (20, 20))

    # Edge detection algorithm
    edges = filters.roberts(image_downscaled)
    edges = edges > 0.1

    # Fill in the edges and fill in holes
    fill_in_edges(edges)
    closed = ndi.binary_fill_holes(edges)

    return image_downscaled, closed


def calculate_volume(im, h):
    # Loop through columns and sum up small disks of volume: 
    # sum of pi * r^2 * dx where dx is one pixel
    volume = 0
    for i in range(len(im[0])):
        col = im[:, i]
        radius = np.sum(col) / len(col) / 2
        volume += np.pi * (radius ** 2)
    # Get maximum height of object
    height = 0
    for i in range(len(im)):
        col = im[i, :]
        if np.sum(col) > (0.5 * len(im[0])):
            first = np.argmax(col == 1)
            last = len(col) - 1 - np.argmax(col[::-1] == 1)
            if last - first > height:
                height = last - first
    # Scale the image back up and calculate the real volume in m^3 of a cubic
    # pixel
    height_pixel = 20 * h / height
    return volume * (height_pixel ** 3)


def show_images(original, downscale, new):
    _, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()

    ax[0].imshow(original, cmap='gray')
    ax[0].set_title('Original Image')

    filtered = np.array([[c if e == 1 else 0 for c, e in zip(r1, r2)] for
                        r1, r2 in zip(downscale, closed)])

    ax[1].imshow(filtered, cmap='gray')
    ax[1].set_title('Downscaled Image')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    name = 'can'

    downscaled, closed = process_image(name)
    # Units are in mililiters (or equivalently cubic centimeters)
    # 0.12 m is roughly half the distance we took our photos away from
    print('Volume:', calculate_volume(closed, 0.12) * 1e6)
    show_images(io.imread(name + '.jpg'), downscaled, closed)
