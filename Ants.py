import numpy as np
import cv2
from matplotlib import pyplot as plt


class Ant:
    def __init__(self, array, position):
        self.array = array
        self.position = position

    @staticmethod
    def is_ant(area, biggest_ant=1000) -> bool:
        return 1 < area < biggest_ant

    @staticmethod
    def is_attached(array, shape) -> bool:
        return np.where(np.logical_and(array, shape.mask))


class Ants:
    def __init__(self, lower=None, upper=None):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.mask = None
        self.ants = []

    def get_mask(self, frame, mask=None):
        # hsv = frame.to_hsv()
        array = frame.array
        if mask is not None:
            array[~mask] = [179, 255, 255]
        self.mask = cv2.inRange(array, self.lower, self.upper)

        # if self.lower[0] == 0:
        #     mask2 = cv2.inRange(hsv,
        #                         np.array([170, 90] + list(self.lower[2:])),
        #                         np.array([179] + list(self.upper[1:])))
        #     self.mask = mask2 + self.mask

        # plt.imshow(frame.to_hsv(array=frame.masked))
        # plt.imshow(frame.to_hsv())
        # plt.imshow(frame.masked)
        # plt.imshow(frame.array)
        # plt.show()

    def count_attached_ants(self, frame, shape):
        gray = cv2.cvtColor(frame.masked, cv2.COLOR_BGR2GRAY)
        # plt.imshow(gray, cmap='gray')
        # plt.show()

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # plt.imshow(blur, cmap='gray')
        # plt.show()

        value, self.mask = cv2.threshold(blur, 85, 255, cv2.THRESH_BINARY_INV)
        # value, self.mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
        # plt.imshow(self.mask)
        number_labels, labels, stats, positions = cv2.connectedComponentsWithStats(self.mask)

        # canny = cv2.Canny(blur, 50, 100)
        # plt.imshow(canny, cmap='gray')
        # plt.show()
        #
        # (cnts, hierachy) = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # rgb = cv2.cvtColor(frame.array, cv2.COLOR_BGR2RGB)
        # cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 2)
        # plt.imshow(rgb)
        # cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # rgb = cv2.cvtColor(frame.array, cv2.COLOR_BGR2RGB)
        # cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 2)
        # plt.imshow(rgb)

        for label in range(number_labels):
            if Ant.is_ant(stats[label][cv2.CC_STAT_AREA]) \
                    and Ant.is_attached(labels == label, shape):
                self.ants.append(Ant(labels == label, positions[label]))
            else:
                self.mask[labels == label] = 0

        for position in [ant.position for ant in self.ants]:
            frame.masked = cv2.circle(frame.masked, np.round(position).astype(int), 1, (0, 0, 255), -1)

        test = cv2.cvtColor(blur.copy(), cv2.COLOR_GRAY2RGB)
        test[self.mask.astype(bool)] = [1, 1, 255]
        plt.imshow(test)

    def dethresh(self, iterations=1):
        for dil_er in range(iterations):
            self.mask = cv2.erode(self.mask.astype(np.uint8), None, iterations=1)
            self.mask = cv2.dilate(self.mask.astype(np.uint8), None, iterations=1)

    def del_ants_far_from_center(self):
        # # only detect ants around the blob
        # number_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_ants)
        #
        # for label in range(number_labels):
        #     # check if blob is close enough
        #     if not radius_pixel * 1 < \
        #            np.linalg.norm((centroids[label] - shape_center[-1])) \
        #            < radius_pixel * 1.3:
        #         mask_ants[mask_ants == label] = 0
        #
        #     else:
        #         mask_ants = self.dethresh(mask_ants, labels, label, stats)
        pass