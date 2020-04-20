import argparse
import matplotlib.pyplot as plt
from random import randint

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression


def open_image(image_path):
    im = Image.open(image_path)
    return im


def load_dots(im, dots_path, random):
    if random:
        w, h = im.size
        step = 100
        x_list, y_list = [], []
        for _ in range(step):
            x_list.append(randint(0, w))
            y_list.append(randint(0, h))
        x_arr = np.asarray(x_list).reshape(-1, 1)
        y_arr = np.asarray(y_list).reshape(-1, 1)
    else:
        data = pd.read_csv(dots_path)
        x_arr = data.iloc[:, 3].values.reshape(-1, 1)
        y_arr = data.iloc[:, 4].values.reshape(-1, 1)
    return x_arr, y_arr


def regression(xs, ys):
    linear_regressor = LinearRegression()
    linear_regressor.fit(xs, ys)
    res_Y = linear_regressor.predict(X)
    return res_Y


def plot_dots_on_image(im, xs, ys, res_Y):
    fig, ax = plt.subplots()
    ax.imshow(im)
    plt.axis([0, im.size[0], 0, im.size[1]])
    plt.scatter(xs, ys)
    plt.plot(xs, res_Y, color='red')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='test.tif', help="Path to the image")
    parser.add_argument('--dots', default='dots.txt', help='Path to x, y coordinate pairs of red dots')
    parser.add_argument('--random', default=False, type=bool, help='Use random dots [true, false]')
    config = parser.parse_args()

    img = open_image(config.image)
    X, Y = load_dots(img, config.dots, config.random)
    Y_pred = regression(X, Y)
    plot_dots_on_image(img, X, Y, Y_pred)


