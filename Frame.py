import imutils
from general import my_screen_width
import numpy as np
import cv2


class Frame:
    def __init__(self, frame_np_array):
        self.array = frame_np_array
        self.masked = np.copy(self.array)

    @staticmethod
    def mouse_callback(event, x, y, image, img):
        # right-click event value is 2
        if event == 2:
            # store the coordinates of the right-click event
            # print('pixel' + str([x, y]))
            print('pixel_value' + str(img[y, x]) + '\n')

            right_clicks.append([x, y])
            # print(len(right_clicks))

    def color_picker(self, windowName='Choose a pixel', array=None, color='hsv'):
        if array is None:
            array = self.array

        img_rgb = imutils.resize(array, width=my_screen_width)
        if color == 'hsv':
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        else:
            img = img_rgb
        cv2.destroyAllWindows()
        cv2.namedWindow(windowName,
                        cv2.WINDOW_NORMAL
                        )
        # cv2.resizeWindow('image', window_width, window_height)

        # set mouse callback function for window
        cv2.setMouseCallback(windowName, self.mouse_callback, img)
        cv2.imshow(windowName, img)
        global right_clicks
        right_clicks = []

        while True:
            k = cv2.waitKey(33)
            if k == 27:  # Esc key to stop
                break
            elif k == -1:  # normally -1 returned,so don't print it
                continue
        cv2.destroyAllWindows()

        return np.array([img[right_click[-1], right_click[0]] for right_click in right_clicks]), img_rgb

    def crop_to_roi(self, array=None, roi_indices=None, message=None):
        if array is None:
            array = self.array
        if roi_indices is None:
            roi_indices = self.find_roi_indices(message=message)

        array = array[int(roi_indices[1]):int(roi_indices[1] + roi_indices[3]),
                      int(roi_indices[0]):int(roi_indices[0] + roi_indices[2])]
        array = cv2.flip(array, 1)
        return array

        # self.masked = self.masked[int(roi_indices[1]):int(roi_indices[1] + roi_indices[3]),
        #               int(roi_indices[0]):int(roi_indices[0] + roi_indices[2])]
        # self.masked = cv2.flip(self.masked, 1)

    def find_roi_indices(self, message=None) -> np.array:
        if message is None:
            message = 'Select ROI where the shape traverses and press Esc'
        selection_roi = cv2.selectROI(message,
                                      imutils.resize(self.array, width=int(my_screen_width / 2)),
                                      )
        cv2.destroyAllWindows()
        roi_indices = [int(roi_i * self.array.shape[1] / my_screen_width * 2) for roi_i in selection_roi]
        print('roi_indices ' + str(roi_indices))
        return roi_indices

    def imshow(self):
        cv2.imshow("Frame", self.masked)

    def resize_frame(self, width):
        self.array = imutils.resize(self.array, width=width)
        self.masked = imutils.resize(self.masked, width=width)

    def to_hsv(self, array=None):
        if array is None:
            array = self.array
        return cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

    def mask_me(self, mask, color=(0, 100, 255)):
        self.masked[mask == 255] = color
