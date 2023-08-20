import classes
import os
import cv2
game = classes.Game(['./pics/tetresized/' + fileName for fileName in os.listdir('./pics/tetresized/')], './pics/Canvas.png')
# print(['./pics/pieces/' + fileName for fileName in os.listdir('./pics/pieces')])
while not game.gameOver:
    game.move()


cv2.waitKey()