from os import path
import cv2
from pandas import read_excel
import numpy as np
my_screen_width = 1900


def movie_address(name):
    df = read_excel('Fixpoint_files.xlsx', engine='openpyxl')
    return [df.loc[df['Name'] == name]['Directory'].values[0] + path.sep + movie
            for movie in df.loc[df['Name'] == name]['Videos'].values]


# tentative hsv values
# lower_ants, upper_ants = ((85, 0, 30), (149, 105, 104))
# lower_shape, upper_shape = ((10, 0, 27), (84, 122, 118))


def flatten(t):
    return [item for sublist in t for item in sublist]

def get_image():
    vs = cv2.VideoCapture(movie_address('large')[3])
    vs.set(cv2.CAP_PROP_POS_FRAMES, 2000)
    img_rgb = vs.read()[1]
    return img_rgb


def buffer_hsv(hsv_values, buffer=0):
    if type(hsv_values) == list:
        hsv_values = np.stack(hsv_values)
    lower = (int(max(0, min(hsv_values[:, 0]) - buffer)),
             int(max(0, min(hsv_values[:, 1]) - buffer)),
             int(max(0, min(hsv_values[:, 2]) - buffer)))

    upper = (int(min(179, max(hsv_values[:, 0]) + buffer)),
             int(min(255, max(hsv_values[:, 1]) + buffer)),
             int(min(255, max(hsv_values[:, 2]) + buffer)))
    return lower, upper


radius_pixel = {'large': 50
                }


lower_ants, upper_ants = ((87, 0, 49), (159, 64, 116))
lower_shape, upper_shape = ((0, 0, 19), (86, 82, 111))
roi = [78, 571, 3648, 875]
start_pt = np.array([3137,  307])

