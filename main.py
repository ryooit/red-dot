import argparse
import math

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import LinearRegression


def load_dots(dots_path):
    data = pd.read_csv(dots_path)
    x_arr = data.iloc[:, 0].values.reshape(-1, 1)
    y_arr = data.iloc[:, 1].values.reshape(-1, 1)
    return x_arr, y_arr


def regression(xs, ys):
    linear_regressor = LinearRegression()
    linear_regressor.fit(xs, ys)
    res_Y = linear_regressor.predict(xs)
    coef = linear_regressor.coef_
    intercept = linear_regressor.intercept_
    print('regressor coef:', coef)
    print('regressor intercept:', intercept)
    return res_Y, coef, intercept


def plot_dots_on_image(xs, ys, res_Y):
    plt.scatter(xs, ys)
    plt.plot(xs, res_Y, color='red')
    plt.show()


def point_to_line_distance(xs, ys, m, b):
    ds = []
    for x, y in zip(xs, ys):
        d = abs(m * x - y + b) / math.sqrt(m ** 2 + 1)
        ds.append(round(d[0][0], 2))
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dots', default='dots.txt', help='Path to x, y coordinate pairs of red dots')
    config = parser.parse_args()

    X, Y = load_dots(config.dots)

    Y_pred, coef, intercept = regression(X, Y)

    df_result = pd.DataFrame({
        'X': X.flatten(),
        'Y': Y.flatten(),
        'distance': point_to_line_distance(X.flatten(), Y.flatten(), coef, intercept)
    })
    print(df_result)
    df_result.to_csv('result.csv')

    plot_dots_on_image(X, Y, Y_pred)





