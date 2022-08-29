from Frame import Frame
import numpy as np
from general import buffer_hsv
import cv2
import imutils
from scipy import ndimage

struct_unit = [[False, True, False], [True, True, True], [False, True, False]]


class Shape:
    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self.mask = None
        self.numcnts = 1

    def pick_HSV(self, frame: Frame):
        hsv_values, img_rgb_roi = frame.color_picker(windowName='Click Tracking Color with right click')
        if self.lower is not None:
            hsv_values = np.vstack([np.array(self.lower), np.array(self.upper), hsv_values])
        self.lower, self.upper = buffer_hsv(hsv_values, buffer=5)
        print(self.lower)
        print(self.upper)

    def get_mask(self, frame):
        hsv = frame.to_hsv()
        mask = cv2.inRange(hsv, np.array(self.lower), np.array(self.upper))

        assert np.any(mask), 'No shape found.'

        # if self.upper[0] > 178:
        #     mask2 = cv2.inRange(frame.to_hsv(),
        #                         np.array([0] + list(self.lower[1:])),
        #                         np.array([3] + list(self.upper[1:])))
        #     mask = mask + mask2

        # if self.upper[0] == 179:
        #     mask2 = cv2.inRange(hsv,
        #                         np.array([1, 90, 140]),
        #                         np.array([10, 120, 160]))
        #     mask = mask2 + mask
        # self.mask = cv2.dilate(mask.astype(np.uint8), None, iterations=1)
        self.mask = mask

    def dethresh(self, iterations=1):
        for dil_er in range(iterations):
            mask = cv2.erode(self.mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)
            self.mask = mask

    def find_contours(self, mask) -> list:
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = list(imutils.grab_contours(cnts))
        cnts = [ci for ci in cnts if cv2.contourArea(ci) > 2]

        if len(cnts) == self.numcnts:
            return cnts
        elif len(cnts) > self.numcnts:
            areas = [cv2.contourArea(ci) for ci in cnts if cv2.contourArea(ci) > 2]
            minArea = sorted(areas)[-self.numcnts - 1]
            return [ci for ci in cnts if cv2.contourArea(ci) > minArea]
        else:
            raise ValueError('Found ' + str(len(cnts)) + ' ccs instead of ' + str(self.numcnts))

        # if len(cnts) > numconts:
        #     print([solidity(ci) for ci in cnts])
        #     minimum = sorted(cnts, key=solidity)[-1]
        #     cnts = [ci for ci in cnts if solidity(ci) > minimum]

    def find_position(self, frame):
        cnts = self.find_contours(self.mask)
        [cX, cY], angle = [np.NaN, np.NaN], np.NaN
        c = np.vstack([cnts[i] for i in range(len(cnts))])
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))

            # fit ellipse
            # _, _, angle = cv2.fitEllipse(c)
            # fit rectangle
            _, _, angle = cv2.minAreaRect(c)

            angle = -((angle + 90) % 360 - 180) % 360
            if not 0 <= angle <= 360:
                print(angle)

            # correct angle: [smaller contour, larger contour]
            if np.argsort([cv2.contourArea(ci) for ci in cnts])[0] == 1:
                cnts.append(cnts[0])
                cnts = cnts[1:]

            cXis = [0] * self.numcnts  # [smaller, larger]
            cYis = [0] * self.numcnts

            for i, ci in enumerate(cnts):
                M = cv2.moments(ci)
                if M["m00"] > 0:
                    cXis[i] = int(M["m10"] / M["m00"])
                    cYis[i] = int(M["m01"] / M["m00"])

            if 135 < angle < 225 and cXis[1] > cXis[0]:
                angle += 180
            elif (0 < angle < 45 or 315 < angle < 360) and cXis[1] > cXis[0]:
                angle += 180
            elif 45 < angle < 135 and cYis[0] > cYis[1]:
                angle += 180
            elif 225 < angle < 315 and cYis[1] > cYis[0]:
                angle += 180
            angle = angle % 360
            angle = np.radians(angle)

            # calculate vector line at angle of bounding box
            length = 300
            P2x = int(cX + length * np.cos(angle))
            P2y = int(cY - length * np.sin(angle))

            # display on Image
            frame.masked = cv2.circle(frame.masked, (int(cX), int(cY)), 5, (0, 0, 255), -1)
            frame.masked = cv2.line(frame.masked, (cX, cY), (P2x, P2y), (255, 255, 255), 2)
            frame.masked = cv2.drawContours(frame.masked, [box], 0, (255, 0, 0), 2)
        return [cX, cY], angle

    @staticmethod
    def dilated_mask(array, iterations=20) -> np.array:
        area_to_search = ndimage.binary_dilation(array, structure=struct_unit, iterations=iterations)
        return area_to_search

    @staticmethod
    def choose_largest_cc(array):
        number_labels, labels, stats, positions = cv2.connectedComponentsWithStats(array.astype(np.uint8))

        cc = np.zeros_like(array)
        for i in range(number_labels)[1:]:
            if 100 < stats[:, cv2.CC_STAT_AREA][i] < array.size * 0.9:
                cc[labels == i] = True
        # biggest_label = np.argsort(stats[:, cv2.CC_STAT_AREA])[-2]
        return cc
