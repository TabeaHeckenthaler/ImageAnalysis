from Frame import Frame
from BackgroundColorDetector import BackgroundColorDetector
from Shape import Shape
from Ants import Ants
from general import my_screen_width, flatten, buffer_hsv

from matplotlib import pyplot as plt
import json
import cv2
from os import path
import os
import glob
import scipy.io
import numpy as np
import imutils
from scipy.signal import medfilt
from tqdm import tqdm


# lower_ants_rgb, upper_ants_rgb = ((65, 30, 34), (134, 97, 104))
# lower_shape_hsv, upper_shape_hsv = [120, 90, 50], [160, 210, 90]


address_tracked = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results{5}Free{6}Output Videos{7}Special T'. \
    format(*[path.sep for _ in range(8)])
address_original1 = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Videos{5}'.format(*[path.sep for _ in range(6)])
address_original2 = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes{5}'.format(*[path.sep for _ in range(6)])
address_video_data_cell = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results{5}video_data_cell.mat'. \
    format(*[path.sep for _ in range(6)])

dilation = {'X': 1, 'L': 3, 'M': 3, 'S': 2}  # XL is denoted as X


def bbox2(img, buffer=0):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [cmin-buffer, rmin-buffer, (cmax-cmin) + 2 * buffer, (rmax-rmin) + 2 * buffer]


class AntCounter:
    def __init__(self, filename, antlength):
        self.filename = filename
        self.vs = cv2.VideoCapture(self.find_original_movie_address())
        self.frames = self.get_frames().tolist()
        self.lower_shape_hsv, self.upper_shape_hsv = None, None
        self.lower_ants_rgb, self.upper_ants_rgb = None, None
        self.choose_threshold_values()
        self.antlength = antlength
        self.roi_indices = self.choose_roi()
        self.bc = None

    def choose_roi(self):
        roi_dict = self.load_roi_dict()

        if self.filename not in roi_dict.keys():
            frame = Frame(self.vs.read()[1])
            roi_dict[self.filename] = frame.find_roi_indices(message='Choose Roi for entire exp, press Esc')
            self.save_roi_dict(roi_dict)

        return roi_dict[self.filename]

    def choose_threshold_values(self):
        value_dict = self.load_value_dict()

        if self.filename not in value_dict.keys():
            rgb_values_ants, hsv_values_humans = self.pick_thresholds()

            value_dict = self.load_value_dict()
            value_dict[self.filename] = [rgb_values_ants, hsv_values_humans]
            self.save_value_dict(value_dict)

        self.lower_ants_rgb, self.upper_ants_rgb = value_dict[self.filename][0]
        self.lower_shape_hsv, self.upper_shape_hsv = value_dict[self.filename][1]

    def pick_thresholds(self):
        rgb_values_ants = []
        hsv_values_shape = []
        # for frames in [self.frames[0], self.frames[-1]]:
        for frames in [self.frames[0]]:
            self.vs.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
            frame = Frame(self.vs.read()[1])
            roi = frame.crop_to_roi(message='Roi of shape with ants, press Esc')
            rgb, img_rgb_roi = frame.color_picker(windowName='ants', color='rgb', array=roi)

            hsv, img_rgb_roi = frame.color_picker(windowName='shape', color='rgb', array=frame.to_hsv(array=roi))
            rgb_values_ants = rgb_values_ants + list(rgb)
            hsv_values_shape = hsv_values_shape + list(hsv)

            self.vs.set(cv2.CAP_PROP_POS_FRAMES, frames[1])
            frame = Frame(self.vs.read()[1])
            roi = frame.crop_to_roi(message='Roi of shape with ants, press Esc')
            rgb, img_rgb_roi = frame.color_picker(windowName='ants', color='rgb', array=roi)

            hsv, img_rgb_roi = frame.color_picker(windowName='shape', color='rgb', array=frame.to_hsv(array=roi))
            rgb_values_ants = rgb_values_ants + list(rgb)
            hsv_values_shape = hsv_values_shape + list(hsv)

        rgb_values_ants = buffer_hsv(rgb_values_ants, buffer=15)
        hsv_values_shape = buffer_hsv(hsv_values_shape, buffer=15)
        return rgb_values_ants, hsv_values_shape

    @staticmethod
    def load_value_dict():
        with open('threshold_values_free_traj_ants.json', 'r') as f:
            value_dict = json.load(f)
        return value_dict

    @staticmethod
    def save_value_dict(value_dict):
        with open('threshold_values_free_traj_ants.json', 'w') as f:
            json.dump(value_dict, f)

    @staticmethod
    def load_roi_dict():
        with open('roi_free_traj_ants.json', 'r') as f:
            value_dict = json.load(f)
        return value_dict

    @staticmethod
    def save_roi_dict(value_dict):
        with open('roi_free_traj_ants.json', 'w') as f:
            json.dump(value_dict, f)

    def original_filename(self):
        return 'S' + self.filename + '.MP4'

    def find_original_movie_address(self):
        files = [glob.glob(path.join(address, "**", self.original_filename()), recursive=True)
                 for address in [address_original1, address_original2]]
        files = flatten(files)
        if len(files) == 1:
            print(files[0])
            return files[0]
        raise ValueError('Where is the original? Could not find ' + self.original_filename())

    def get_frames(self):
        mat = scipy.io.loadmat(address_video_data_cell)['video_data_cell']
        index = np.where(mat[:, 0] == [self.original_filename()[:-4]])
        assert len(index) == 1, 'Didnt find ' + self.original_filename()[:-4] + 'in video data cell'
        return mat[:, 1][index][0]

    def cut_out_shape_area(self, frame):
        frame.array = frame.crop_to_roi(array=frame.array, roi_indices=self.roi_indices)
        frame.masked = frame.crop_to_roi(array=frame.masked, roi_indices=self.roi_indices)

        # rgb_values_ants, img_rgb_roi = frame.color_picker(color='rgb')
        # print(buffer_hsv(rgb_values_ants, buffer=10))

        shape = Shape(lower=self.lower_shape_hsv, upper=self.upper_shape_hsv)
        shape.get_mask(frame)
        shape.dethresh(iterations=1)
        area_to_search = shape.choose_largest_cc(shape.mask).astype(bool)
        frame.masked[area_to_search] = (50, 255, 255)
        area_to_search = shape.dilated_mask(area_to_search, iterations=self.antlength)
        bbox2_ind = bbox2(area_to_search, buffer=0)

        frame.array = frame.crop_to_roi(array=frame.array, roi_indices=bbox2_ind)
        frame.masked = frame.crop_to_roi(array=frame.masked, roi_indices=bbox2_ind)
        area_to_search = frame.crop_to_roi(array=area_to_search.astype(np.uint8), roi_indices=bbox2_ind)
        shape.mask = frame.crop_to_roi(array=shape.mask, roi_indices=bbox2_ind)

        blur = cv2.GaussianBlur(frame.array, (41, 41), 0)
        # plt.imshow(blur, cmap='gray')
        # plt.show()

        if self.bc is None:
            self.bc = self.find_background_color(blur)

        frame.masked[cv2.dilate(shape.mask.astype(np.uint8), None, iterations=dilation[size[0]]).astype(bool)] = self.bc
        frame.masked[~area_to_search.astype(bool)] = self.bc
        # plt.imshow(frame.array, cmap='gray')
        # plt.show()
        return shape

    def find_background_color(self, img):
        bcd = BackgroundColorDetector(img)
        return bcd.detect()

    def count_ants(self, frames, display=False):
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, int(self.vs.get(cv2.CAP_PROP_FRAME_COUNT) * 1 / 3))

        n_ants = []
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        for i in tqdm(range(*frames)):
            frame = Frame(self.vs.read()[1])
            shape = self.cut_out_shape_area(frame)
            # plt.imshow(frame.masked)
            # plt.show()

            ants = Ants()
            ants.count_attached_ants(frame, shape)
            n_ants.append(len(ants.ants))
            frame.masked = cv2.putText(frame.masked, str(n_ants[-1]),
                                       (frame.masked.shape[0]//20, frame.masked.shape[0]//10), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.4, (10, 10, 10), 1, cv2.LINE_AA)
            if display:
                cv2.imshow(self.filename, imutils.resize(frame.masked, width=my_screen_width//4))

            key = cv2.waitKey(1) & 0xFF
            # print(n_ants[-1])

            if frame is None:
                break

            if i == frames[0] or i == frames[1]:
                path_save = path.join('control_images', self.filename + "_" + str(i) + '_' + str(i) + '.png')
                self.save_snapshot(frame.masked, path_save=path_save)

        return n_ants

    def save_snapshot(self, img, path_save=None):
        if path_save is None:
            path_save = path.join('control_images', self.filename)
        cv2.imwrite(path_save, img)

    def save_data(self, n_ants: list, appendix: str, path_save=None):
        if path_save is None:
            path_save = path.join('ant_counting_results', self.filename + '_' + appendix + '.json')
        with open(path_save, 'w') as f:
            json.dump(n_ants, f)

    def load_counting_results(self, appendix: str, path_save=None):
        if path_save is None:
            path_save = path.join('ant_counting_results', self.filename + '_' + appendix + '.json')
        with open(path_save, 'r') as f:
            n_ants_saved = json.load(f)
        return n_ants_saved

    def plot_n_ants(self, n_ants, ax):
        ax.plot(n_ants)


if __name__ == '__main__':
    all_files = os.listdir(address_tracked)
    movie_files = {}
    antlength = {'LS': 10, 'MS': 10, 'SS': 15, 'XL': 8}

    for a in all_files:
        if a[:2] not in movie_files.keys():
            movie_files[a[:2]] = []
        movie_files[a[:2]].append(a)

    movies = {
        'MS': ['4410001_freeMSpecialT'],
        'LS': ['4210007_freeLSpecialT'],
        'SS': ['4210009_freeSSpecialT'],
        'XL': ['4280003_freeXLSpecialT'],
              }

    # file = '4410001_freeMSpecialT'
    # ant_counter = AntCounter(file, antlength['MS'])

    # frames = [6313, 10427]
    # n_ants = ant_counter.count_ants(frames)

    # movie_files = {'XL': ['XLSPT_4290001_freeXLSpecialT_7']}

    fig, axs = plt.subplots(4)
    sizes_filter_window = {'XL': 11, 'LS': 5, 'MS': 11, 'SS': 5, }

    for size, ax in zip(sizes_filter_window, axs):
        # movies = set(['_'.join(filename.split('_')[1:3]) for filename in movie_files[size]])

        for file in movies[size]:
            print(file)
            ant_counter = AntCounter(file, antlength[size])
            for i, frames in enumerate(ant_counter.frames):
                # n_ants = ant_counter.count_ants(frames)
                # n_ants = ant_counter.count_ants([7177, 7577], display=True)
                # n_ants2 = ant_counter.count_ants([frames[1] - 1000, frames[1] - 700], display=True)
                # DEBUG = 1

                n_ants = ant_counter.load_counting_results(str(i))
                # n_ants_filtered = ant_counter.load_counting_results(str(i) + '_filtered')
                n_ants_filtered = medfilt(n_ants, sizes_filter_window[size] * 2 + 1).tolist()
                # if len(n_ants) > 6880:
                #     DEBUG = 1
                ant_counter.plot_n_ants(n_ants_filtered, ax)
                # n_ants = n_ants[:6900]

                # DEBUG = 1
                #
                # ant_counter.save_data(n_ants, str(i))
                ant_counter.save_data(n_ants_filtered, str(i) + '_filtered')
    plt.show()

