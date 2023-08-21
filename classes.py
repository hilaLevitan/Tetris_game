import cv2
import random as rand
import numpy as np
import math
import screeninfo


class Piece:
    def __init__(self, piecePath):
        self.piecePath = piecePath
        self.image = cv2.imread(f'{piecePath}', cv2.IMREAD_UNCHANGED)
        self.pieceLocation = np.array([0, 0])
        self.pieceValocity = np.array([30, 0])
        self.mask = self.getMask()

    def getMask(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    def draw(self, canvasImage):
        height, width = self.image.shape[:2]
        for y in range(self.pieceLocation[0], self.pieceLocation[0] + height):
            for x in range(self.pieceLocation[1], self.pieceLocation[1] + width):
                overlay_color = self.image[y - self.pieceLocation[0], x - self.pieceLocation[1],
                                :3]  # first three elements are color (RGB)
                overlay_alpha = self.image[
                                    y - self.pieceLocation[0], x - self.pieceLocation[
                                        1], 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0
                # get the color from the background image
                background_color = canvasImage[y, x]
                # combine the background color and the self.image color weighted by alpha
                composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha
                # update the background image in place
                canvasImage[y, x] = composite_color

    def rotate_image_2(self, angle, topLeftX, topLeftY):
        height, width = self.image.shape[:2]

        # Calculate the rotation matrix
        image_center = (topLeftX + width / 2, topLeftY + height / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # Perform the rotation
        result = cv2.warpAffine(self.image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)

        # Calculate the rotated top-left corner
        rotated_top_left = np.dot(rot_mat, np.array([topLeftX, topLeftY, 1]))
        new_top_left_x = int(rotated_top_left[0])
        new_top_left_y = int(rotated_top_left[1])
        self.image=result
        self.mask=self.getMask()
        self.pieceLocation=np.array([ new_top_left_x, new_top_left_y])


    def rotate_image(self, angle):
        # Calculate new dimensions to fit the rotated image
        height, width = self.image.shape[:2]
        new_width = int(np.abs(width * np.cos(np.radians(angle))) + np.abs(height * np.sin(np.radians(angle))))
        new_height = int(np.abs(width * np.sin(np.radians(angle))) + np.abs(height * np.cos(np.radians(angle))))
        # Calculate the center of the original image
        image_center = (width // 2, height // 2)
        # Calculate the rotation matrix
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # Adjust the translation in the matrix to center the image
        rot_mat[0, 2] += (new_width - width) / 2
        rot_mat[1, 2] += (new_height - height) / 2

        # Perform the rotation
        result = cv2.warpAffine(self.image, rot_mat, (new_width, new_height), flags=cv2.INTER_LINEAR)
        if result.shape[1]<self.image.shape[1]:
            self.pieceLocation[1]+=int(self.image.shape[1]/2-result.shape[1]/2)
        else:
            self.pieceLocation-=int(result.shape[1]/2-self.image.shape[1]/2)
        self.image=result
        self.mask = self.getMask()
    def move(self,key,canvasImage):
        if key == ord('a'):
            if (self.pieceLocation[1] - 10 >= 0):
                self.pieceLocation[1] -= 10
        elif key == ord('d'):
            if (self.pieceLocation[1] + 10 < canvasImage.shape[1] - self.image.shape[1]):
                self.pieceLocation[1] += 10
        elif key == ord('s'):
            self.pieceLocation[0] += 10
        elif key== ord('w'):
            self.rotate_image(90)
        self.pieceLocation += self.pieceValocity
    def __copy__(self):
        newPiece = Piece(self.piecePath)
        return newPiece


class Game:
    def __init__(self, listOfPices, canvasPath):
        self.score=0
        self.score_per_piece=10
        self.gameOver = False
        self.pieces = listOfPices
        self.currPiece = None
        self.canvasImage_orig = cv2.imread(f"{canvasPath}")
        self.curr_canvasImage = self.canvasImage_orig.copy()
        cv2.namedWindow("canvas", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.setNewPiece()

    def setNewPiece(self):
        self.currPiece: Piece = Piece(rand.choice(self.pieces))
        # set location of the new piece to [0, random location on the x axis
        self.currPiece.pieceLocation = np.array(
            [0, rand.randint(0, self.curr_canvasImage.shape[1] - self.currPiece.image.shape[1])])

    def checkCollision(self):
        for i in range(self.currPiece.pieceLocation[1],
                       self.currPiece.pieceLocation[1] + self.currPiece.image.shape[1]):
            for j in range(self.currPiece.pieceLocation[0],
                           self.currPiece.pieceLocation[0] + self.currPiece.image.shape[0]):
                if self.currPiece.mask[j-self.currPiece.pieceLocation[0],i-self.currPiece.pieceLocation[1]]==255:
                    continue
                if not ((self.curr_canvasImage[self.currPiece.pieceLocation[0] + self.currPiece.image.shape[0], i] ==
                         self.canvasImage_orig
                         [self.currPiece.pieceLocation[0] + self.currPiece.image.shape[0], i]).all()):
                    return True
        return False

    def move(self):
        key = cv2.waitKey(20)
        self.currPiece.move(key=key,canvasImage=self.curr_canvasImage)
        canvasImage: np.ndarray = self.curr_canvasImage.copy()
        self.currPiece.draw(canvasImage)

        cv2.putText(canvasImage,f'score: {self.score}',org = (100,100),fontScale=3,fontFace=5, color = (255,255,255),thickness = 2,lineType=cv2.LINE_AA)
        cv2.imshow('canvas', canvasImage)
        isReachedEndOfCanvas = self.currPiece.pieceLocation[0] + self.currPiece.image.shape[0] > self.curr_canvasImage.shape[
                                   0] - self.currPiece.pieceValocity[0]
        if isReachedEndOfCanvas or self.checkCollision():
            self.score+=self.score_per_piece
            # self.curr_canvasImage=canvasImage
            self.currPiece.draw(canvasImage=self.curr_canvasImage)
            self.setNewPiece()
            self.checkGameStatus()

    def checkGameStatus(self):
        for i in range(0, self.canvasImage_orig.shape[1]):
            if not ((self.curr_canvasImage[200, i] == self.canvasImage_orig
            [200, i]).all()):
                self.gameOver = True
