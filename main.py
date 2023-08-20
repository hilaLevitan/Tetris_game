import classes
import os
import cv2
game = classes.Game(['./pics/pieces/' + fileName for fileName in os.listdir('./pics/pieces/')], './pics/Canvas.png')
# print(['./pics/pieces/' + fileName for fileName in os.listdir('./pics/pieces')])
while not game.gameOver:
    game.move()


cv2.waitKey()