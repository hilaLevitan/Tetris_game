import numpy as np
import cv2
import os
import random

canvasPath = './pics/Canvas.png'
piecePath = './pics/tet/blue.png'

canvasImage_orig = cv2.imread(canvasPath)
pieceImage = cv2.imread(piecePath)
                                # 1080,1920,3
height, width, channels = pieceImage.shape
pieceLocation = np.array([0, int(canvasImage_orig.shape[1]/2)]) # Top Left Corner
pieceVelocity = np.array([1, 0])

isReachedEndOfCanvas = False

while not isReachedEndOfCanvas:
    canvasImage = canvasImage_orig.copy()
    canvasImage[pieceLocation[0]:pieceLocation[0] + height,
                pieceLocation[1]:pieceLocation[1] + width,:] = pieceImage

    cv2.imshow('canvas', canvasImage)
    key = cv2.waitKey(20)
    pieceLocation = pieceLocation + pieceVelocity
    if key == ord('a'):
        pieceLocation[1] -= 10
    elif key == ord('d'):
        pieceLocation[1] += 10
    elif key == ord('s'):
        pieceLocation[0] += 10

    isReachedEndOfCanvas = pieceLocation[0] + height > canvasImage.shape[0] - 10

cv2.waitKey()

