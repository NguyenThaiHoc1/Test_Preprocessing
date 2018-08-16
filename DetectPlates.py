
import cv2
import numpy as np
import math
import Main
import random
import sys
import os

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.7
PLATE_HEIGHT_PADDING_FACTOR = 1.5

MAX_OVERLAP_RATIO, MIN_OVERLAP_RATIO = 1, 0.8
MAX_RATIO, MIN_RATIO = 1.5, 0.5
MAX_ANGLE_DIFF, MIN_ANGLE_DIFF = 30, -30
SHAPE_OF_POSSIBLE_PLATE = (120, 96)

listOfPossiblePlates =[]


def detectPlatesInScene(imgOriginalScene, location):
    listOfRawPossiblePlates = []                   

    height, width, numChannels = imgOriginalScene.shape

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: 
        cv2.imshow("0", imgOriginalScene)

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         # preprocess to get grayscale and threshold images

    if Main.showSteps == True: # show steps #######################################################
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)


    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
    listOfPossibleCharsInScene.sort(key = lambda Char: Char.intCenterX)

    if Main.showSteps == True: # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(len(listOfPossibleCharsInScene)))         # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)

    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(len(listOfListsOfMatchingCharsInScene)))    # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

        cv2.imshow("3", imgContours)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
        possiblePlate = extractPlate(listOfMatchingChars)         # attempt to extract plate
        listOfRawPossiblePlates.append(possiblePlate)                  # add to list of possible plates


    listOfPossiblePlates = groupPossiblePlates(imgOriginalScene, listOfRawPossiblePlates)
    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")          # 13 with MCLRNF1 image

    if Main.showSteps == True: # show steps #######################################################
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)

    if Main.save == True:
        for i in range(0, len(listOfPossiblePlates)):
            # save plate
            fileName = location.split("/")[-1].split(".")
            plateFolder = "outputs/" + fileName[0]

            if not os.path.isdir(plateFolder):
                os.makedirs(plateFolder)
            extractedPlateName = fileName[0] + "/plate_" + str(i) + "." + fileName[1]
            resized_plate = cv2.resize(listOfPossiblePlates[i].imgPlate, SHAPE_OF_POSSIBLE_PLATE)
            cv2.imwrite("outputs/" + extractedPlateName, resized_plate)

    return listOfPossiblePlates