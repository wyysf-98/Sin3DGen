import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm

def read_image_as_grayscale(path):
    ref = cv2.imread(path)
    return cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)


def calculate_images_diversity(ref_dir, image_dirs):
    """To quantify the diversity of the generated images,
        for each training example we calculated the standard devia-
        tion (std) of the intensity values of each pixel over 100 gen-
        erated images, averaged it over all pixels, and normalized
        by the std of the intensity values of the training image.
    """
    all_diversities = []

    for ref_filename in tqdm(os.listdir(ref_dir)):
        if 'png' not in ref_filename:
            continue
        ref_gray = read_image_as_grayscale(os.path.join(ref_dir, ref_filename))
        msk = ref_gray <= 255

        images = []
        for i, images_dir in enumerate(image_dirs):
            try:
                images.append(read_image_as_grayscale(os.path.join(images_dir, ref_filename)))
            except:
                images.append(images[-1])
        images = np.stack(images)

        diversity = np.std(images, axis=0)[msk].mean() / np.std(ref_gray[msk])
        all_diversities.append(diversity)

    return np.mean(all_diversities)

