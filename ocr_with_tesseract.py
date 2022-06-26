""" Dependancies :
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
pip install pytesseract
pip install fuzzywuzzy
"""
import logging
import cv2
import numpy
import pytesseract
import math
import numpy as np
import re, sys, os
import unicodedata
from skimage import color, restoration, img_as_ubyte
from fuzzywuzzy import fuzz
from numpy.random import choice
import time

g = 1
alpha = 1.2  # Simple contrast control
beta = 20  # Simple brightness control
confThreshold = 0.1
nmsThreshold = 0.4
inpWidth = 640
inpHeight = 640


class OnboardingScore:
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.frame = None
        self.associated_name_status = {}
        self.associated_name_fuzzy_score = {}
        self.config = "-l eng --oem 1 --psm 7"
        self.boxFace = None
        self.fuzzy_threshold = 70
        self.face_on_facescan_detected = {}
        self.insightface_model = None
        models_path = '/home/prabodh/personal_space/How_I_Am_Learning_ML/Model_Hub/'
        # Load network

        def get_eastnet(readpath):
            # print readpath
            net = cv2.dnn.readNet(readpath)
            outNames = []
            outNames.append("feature_fusion/Conv_7/Sigmoid")
            outNames.append("feature_fusion/concat_3")
            return net, outNames

        self.East_net, self.outNames = get_eastnet(models_path + 'frozen_east_text_detection.pb')

    def hasNumbers(self, inputstring):
        return bool(re.search(r'\d', inputstring))

    def deskew_patch(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)

        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def deskew_angle(self, image, angle):
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated, angle

    def grouper(self, iterable, interval=2):
        prev = None
        group = []
        for item in iterable:
            if not prev or abs(item[1] - prev[1]) <= interval:
                group.append(item)
            else:
                yield group
                group = [item]
            prev = item
        if group:
            yield group

    def decode(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []
        rects = []
        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                endX = int(offsetX + (cosA * x1_data[x]) + (sinA * x2_data[x]))
                endY = int(offsetY - (sinA * x1_data[x]) + (cosA * x2_data[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences, rects]

    def getboundingbox_for_text_regions(self, indices, rects, boxes, paddingratio=None):
        BBXs = []
        heights = []
        height_ = self.frame.shape[0]
        width_ = self.frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)
        # loop over the bounding boxes
        for i in indices:
            (startX, startY, endX, endY) = rects[i[0]]
            heights.append(endY - startY)
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            if paddingratio:
                dX = int((endX - startX) * paddingratio)
                dY = int((endY - startY) * paddingratio)

                # apply padding to each side of the bounding box, respectively
                startX = max(0, startX - dX)
                startY = max(0, startY - dY)
                endX = min(inpWidth, endX + (dX * 2))  # need to discard2??
                endY = min(inpHeight, endY + (dY * 2))
            dummy = boxes[i[0]]
            BBXs.append([startX, startY, endX, endY, dummy[2]])
        return BBXs, heights

    def generate_text_matching_score(self, str1, str2):
        Ratio = fuzz.ratio(str1.lower(), str2.lower())
        return Ratio

    def check_in_match_name_list(self, det_text, match_list):
        score = []
        matched_text = []

        def split(s):
            temp_s = ""
            for ch in s:
                if ch.isspace():
                    if temp_s:
                        yield temp_s
                        temp_s = ""
                else:
                    temp_s += ch
            if temp_s:
                yield temp_s

        det_text = list(split(det_text))
        filter_text = [txt.split('\n') for txt in det_text]
        det_filter_txt = []
        for txt in filter_text:
            for t in txt:
                det_filter_txt.append(t)

        for text1 in det_filter_txt:
            for idx, text2 in enumerate(match_list):
                ratio = self.generate_text_matching_score(text1, text2)

                if ratio > self.fuzzy_threshold:
                    self.associated_name_status[text2] = True
                    matched_text.append(text1)
                    score.append(ratio)
                    self.associated_name_fuzzy_score[text2].append(ratio)
                if text2 in text1:
                    self.associated_name_status[text2] = True
                    matched_text.append(text1)
                    score.append(100)
                    self.associated_name_fuzzy_score[text2].append(100)

        return matched_text, score

    def feature_from_list(self, image_list, is_facescan=False):
        """This function prepares a list of facial features corresponding to a list of face images.
        :param image_list: List of face images
        :return: List of facial features"""
        feature_list = []
        for idx, image in enumerate(image_list):
            if image.size == 0:
                feature_list.append(None)
            else:
                img = self.insightface_model.get_input(image)
                f = None
                if img is not None:
                    # face is detected by insightface model
                    f = self.insightface_model.get_feature(img)
                    if is_facescan:
                        self.face_on_facescan_detected[idx] = True
                elif is_facescan:
                    self.face_on_facescan_detected[idx] = False
                feature_list.append(f)
        return feature_list

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def select_filter_layers(self, layer, img):
        if layer == 'layer_1':
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        elif layer == 'layer_2':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif layer == 'layer_3':
            img = self.adjust_gamma(img, gamma=1)
        elif layer == 'layer_4':
            """Restoration"""
            psf = np.ones((4, 4)) / 16  # depends on bbx
            img = color.rgb2gray(img)
            img, _ = restoration.unsupervised_wiener(img, psf)
            img = img_as_ubyte(img)
        elif layer == 'layer_5':
            self.select_filter_layers('layer_1', img)
            self.select_filter_layers('layer_4', img)
            self.select_filter_layers('layer_3', img)
        return img

    def verify_text_on_image(self, image_paths, match_name_list, fuzzy_threshold=70, debug=False):
        """ This function returns: score of id card content analysis and face verification."""
        st = time.time()

        self.fuzzy_threshold = fuzzy_threshold

        for i in match_name_list:
            self.associated_name_status[i] = False
            self.associated_name_fuzzy_score[i] = []

        for image_path in image_paths:
            try:
                self.frame = cv2.imread(image_path)
                frame_show = self.frame.copy()
                # Create a 4D blob from frame.
                blob = cv2.dnn.blobFromImage(self.frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True,
                                             False)

                # Run the model
                self.East_net.setInput(blob)
                outs = self.East_net.forward(self.outNames)

                # Get scores and geometry
                scores = outs[0]
                geometry = outs[1]

                [self.boxes, confidences, self.rects] = self.decode(scores, geometry, confThreshold)
                self.indices = cv2.dnn.NMSBoxesRotated(self.boxes, confidences, confThreshold, nmsThreshold)

                values = [v for v in self.associated_name_status.values()]
                if not values.count(False) > 0:
                    continue

                config = "-l eng --oem 1 --psm 1"
                tess_text = get_text_from_image(self.frame, config)
                matched_text, score = self.check_in_match_name_list(tess_text, match_name_list)
                if debug:
                    print("[Info] Output is for Entire Image: \n", tess_text)
                    print('[Info] Matched Text: {} Match Score: {} '.format(matched_text, score),
                      self.associated_name_status)

                # Need to check on different padding size.
                BBXs, heights = self.getboundingbox_for_text_regions(self.indices, self.rects, self.boxes,
                                                                     paddingratio=0.10)
                if heights:
                    heights = sorted(heights)  # Sort heights
                    median_height = heights[int(len(heights) / 2)] / 2  # Find half of the median height
                    BBXs = np.asanyarray(BBXs)
                    bbsortedx = BBXs[np.lexsort((BBXs[:, 2], BBXs[:, 0], BBXs[:, 3], BBXs[:, 1]))]
                    bbsortedx = bbsortedx.tolist()
                    try:
                        combined_bboxes = self.grouper(bbsortedx, median_height)  # Group the bounding boxes
                        g = 0
                        for group_no, group in enumerate(combined_bboxes):
                            # print group
                            g = g + 1
                            x_min = int(min(group, key=lambda k: k[0])[0])  # Find min of x1
                            x_max = int(max(group, key=lambda k: k[2])[2])  # Find max of x2
                            y_min = int(min(group, key=lambda k: k[1])[1])  # Find min of y1
                            y_max = int(max(group, key=lambda k: k[3])[3])  # Find max of y2
                            angle_max = max(group, key=lambda k: k[4])[4]  # Find max of y2
                            roi = self.frame[y_min:y_max, x_min:x_max]
                            if debug:
                                cv2.rectangle(frame_show, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                            for i in range(1, 6):
                                values = [v for v in self.associated_name_status.values()]
                                if values.count(False) > 0:
                                    filtered_roi = self.select_filter_layers('layer_{}'.format(i), roi)
                                    tess_text = get_text_from_image(filtered_roi, self.config)
                                    if debug:
                                        print("Matching with line no. {} layer_{}: ".format(group_no, i), tess_text)

                                    matched_text, score = self.check_in_match_name_list(tess_text, match_name_list)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(e, exc_type, fname, exc_tb.tb_lineno)

                    # print "Matched_text and score: ", recog_text, recog_score

                if debug:
                    cv2.imshow('OCR', frame_show)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.info('Error: {0} {1} {2} {3}'.format(e, exc_type, fname, exc_tb.tb_lineno))

        id_text_score = {}
        for key, values in self.associated_name_fuzzy_score.items():
            if values:
                id_text_score[key] = max(values)
            else:
                id_text_score[key] = 0
        return self.associated_name_status, id_text_score


def get_text_from_image(image, config):
    tess_text = pytesseract.image_to_string(image, config=config)
    if type(tess_text) != str:
        tess_text = unicodedata.normalize('NFKD', tess_text).encode('ascii', 'ignore')
    return tess_text


if __name__ == '__main__':
    import glob
    data_tobe_verified = ['APPLICANT', 'NAME', 'INCOMETAX']
    image_paths = ['ABCD_pan_card.jpg']
    onboarding = OnboardingScore()
    # print id_content_analysis
    result = onboarding.verify_text_on_image(image_paths, match_name_list=data_tobe_verified, debug=True)
    print(result)

