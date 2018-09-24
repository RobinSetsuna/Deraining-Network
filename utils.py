from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


# -----------------------------
# new added functions for pix2pix

def load_data(image_path, fine_size, is_test=False):
    img_a, img_b = load_image(image_path)
    img_a, img_b = preprocess(img_a, img_b, is_test=is_test, fine_size=fine_size)

    # 某种图像预处理方式
    img_a = img_a / 127.5 - 1.
    img_b = img_b / 127.5 - 1.

    # img_ab大小: (fine_size, fine_size, input_c_dim + output_c_dim)
    img_ab = np.concatenate((img_a, img_b), axis=2)

    return img_ab


def load_image(image_path):
    input_img = imread(image_path)
    # 存储模式 高X宽X深度
    w = int(input_img.shape[1])
    w2 = int(w / 2)
    # 将图片从中部切分，A为左图，B为右图
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B


def preprocess(a, b, fine_size, is_test=False, load_size=286):
    # 如果是测试
    #if is_test:
    a = scipy.misc.imresize(a, [fine_size, fine_size])
    b = scipy.misc.imresize(b, [fine_size, fine_size])
    """else:
        a = scipy.misc.imresize(a, [load_size, load_size])
        b = scipy.misc.imresize(a, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))

        a = a[h1:h1 + fine_size, w1:w1 + fine_size]
        b = b[h1:h1 + fine_size, w1:w1 + fine_size]"""

    return a, b


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.
