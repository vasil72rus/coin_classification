import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance


def get_edges(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    enhancer = ImageEnhance.Contrast(Image.fromarray(img))

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(f'Original')
    plt.xticks([])
    plt.yticks([])
    for i, j in zip(np.arange(1, 4, 0.5), range(1, 6)):
        factor = i
        im_output = enhancer.enhance(factor)
        im_output = im_output.convert()
        im_output = np.array(im_output)

        clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(5, 5))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l2 = clahe.apply(l)

        lab = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        sobelX = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, borderType=border)
        sobelY = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, borderType=border)

        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))

        sobelCombined = cv2.bitwise_or(sobelX, sobelY)
        plt.subplot(2, 3, j + 1)
        plt.imshow(sobelCombined)
        plt.title(f'Contrast = {i}')
        plt.xticks([])
        plt.yticks([])

    plt.show()
