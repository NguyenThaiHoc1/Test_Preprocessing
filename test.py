import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import PossibleChar
import DetectChars

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9


def preprocess(imgOriginal):
    print ("this shape of img Original: {0}".format(imgOriginal.shape))

    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    cv2.imwrite("./images/0a.png", imgHue)

    cv2.imwrite("./images/0b.png", imgSaturation)

    cv2.imwrite("./images/0c.png", imgValue)

    return imgValue

def maximizeContrast(imgGrayscale):
    print ("this shape of img GrayScale: {0}".format(imgGrayscale.shape))
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    cv2.imwrite("./images/0d-1.png", imgTopHat)
    cv2.imwrite("./images/0d-2.png", imgBlackHat)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    cv2.imwrite("./images/0d-3.png", imgGrayscalePlusTopHat)

    cv2.imwrite("./images/0d-4.png", imgGrayscalePlusTopHatMinusBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        cv2.drawContours(imgContours, contours, i, SCALAR_WHITE)

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
            listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars

    print("\nstep 2 - len(contours) = " + str(len(contours)))                       # 2362 with MCLRNF1 image
    print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))       # 131 with MCLRNF1 image
    cv2.imwrite("./images/2b.png", imgContours)
    print (intCountOfPossibleChars)
    return listOfPossibleChars



def detectPlatesInScene(imgOriginalScene, location):
    listOfRawPossiblePlates = []                   

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    cv2.imwrite("./images/0.png", imgOriginalScene)

    imgGrayscaleScene, imgThreshScene = preprocess(imgOriginalScene)         # preprocess to get grayscale and threshold images

    cv2.imwrite("./images/1a.png", imgGrayscaleScene)
    cv2.imwrite("./images/1b.png", imgThreshScene)

    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)



image = cv2.imread("./plate.jpg", 1)
detectPlatesInScene(image, "./1.png")




