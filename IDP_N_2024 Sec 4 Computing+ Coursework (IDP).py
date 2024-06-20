# IMPORTANT THINGS TO NOTE:
# Hats can merge and cannot be unmerged, just restart the program.
# All the hats should be in the same folder as the python file.
# Hats are unrendered when no facial mesh is detected.

import cv2 # Have to install this module using pip3 install opencv-python dlib imutils.
import cvzone # Install with pip3 install cvzone, documentation found here: https://github.com/cvzone/cvzone/blob/master/README.md

from cvzone.FaceMeshModule import FaceMeshDetector # Face Mesh detector from cvzone.
from cvzone.HandTrackingModule import HandDetector # Hand Tracker from cvzone.
from cvzone.Utils import rotateImage # rotateImage function from cvzone.

# Necessary math module for angle calculation and resizing.
import math
from math import *
from math import hypot


# Angle Detector math.
def angle_truncation(angle):
    while angle < 0.0:
        angle += pi * 2
    return angle

# Angle Detector function.
def checkAngle(xyog, xylandmark):
    # Finding distance between two points that is used to get the angle.
    deltaY = xylandmark[1] - xyog[1]
    deltaX = xylandmark[0] - xyog[0]
    # Function returns angle in degrees.
    return math.degrees(angle_truncation(atan2(deltaY, deltaX))) * -1

# Initialize the webcam and set it to the third camera (index 2)
cap = cv2.VideoCapture(0)
# Set the camera quality
cap.set(3, 1280)
cap.set(4, 720)


# Input Roblox Hat Images by reading the file names. Files can be found in the folder along with the code.
strawHatImage = cv2.imread("StrawHat.png",cv2.IMREAD_UNCHANGED)
topHatImage = cv2.imread("TopHat.png",cv2.IMREAD_UNCHANGED)
partyHatImage = cv2.imread("PartyHat.png",cv2.IMREAD_UNCHANGED)
vikingHatImage = cv2.imread("VikingHat.png",cv2.IMREAD_UNCHANGED)
crownHatImage = cv2.imread("CrownHat.png",cv2.IMREAD_UNCHANGED)
TinHatImage = cv2.imread("TinHat.png",cv2.IMREAD_UNCHANGED)
BaseballHatImage = cv2.imread("BaseballHat.png",cv2.IMREAD_UNCHANGED)
DominoHatImage = cv2.imread("DominoHat.png",cv2.IMREAD_UNCHANGED)
RobotHatImage = cv2.imread("RobotHat.png",cv2.IMREAD_UNCHANGED)

# Hat class to make adding extra hats easier
class DragHat():
    # Initialising of the class variables
    def __init__(self, hat, posCenter, boundarySize = [200,200]):
        # self.hat is the hat image itself.
        self.hat = hat
        # self.posCenter is the position of the hat.
        self.posCenter = posCenter
        # self.boundarySize is the collision boundary of the hat which will affect how it interacts with the hand and face mesh.
        self.boundarySize = boundarySize
    # Updating class variables (used for updating hat position)
    def update(self, position):
        # Updating hand position.
        self.posCenter = position[0], position[1]

# Appending all the hats input previously so that they will be rendered in frame.
hatList = []
hatList.append(DragHat(strawHatImage, [200, 700]))
hatList.append(DragHat(topHatImage, [400, 700]))
hatList.append(DragHat(partyHatImage, [600, 700]))
hatList.append(DragHat(vikingHatImage, [800, 700]))
hatList.append(DragHat(crownHatImage, [1000, 700]))
hatList.append(DragHat(TinHatImage, [1200, 700]))
hatList.append(DragHat(RobotHatImage, [200, 400]))
hatList.append(DragHat(DominoHatImage, [400, 400]))
hatList.append(DragHat(BaseballHatImage, [600, 400]))


# Initialize the FaceMeshDetector class with the given parameters
detector = FaceMeshDetector(staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5)

# Initialize the HandDetector class with the given parameters
handDetector = HandDetector(detectionCon=0.7)

# Any variables needed to be intialised here before the loop begins, otherwise would throw an error.
attached = False


# Loop to continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    # Flip image for easier time navigating the hands.
    img = cv2.flip(img, 1)

    # Find the human face and hand in frame.
    # Set draw to True if you want to see the mesh for the hand and face.
    # img2 is used so that the rendered hats do not interfere with the detection of faces and hands
    img2, faces = detector.findFaceMesh(img, draw=False)
    hand, img = handDetector.findHands(img, flipType = False,draw=False)

    # Initialising the landmark lists for every face detected.
    for face in faces:
        lmlist = face
        
        # Check if any body landmarks are detected, otherwise code will not run.
        if lmlist:
            # Check if any hands are detected in frame, if not, it will not crash since there is try and except catch.
            try:
                # Landmark list for hand
                lmList = hand[0]["lmList"]
                # Finding distance between index finger and thumb for pinch. l is that distance, lmList[8][:2] is the index finger coordinates, lmList[4][:2] is the thumb coordinates.
                l, _ , _ = handDetector.findDistance(lmList[8][:2], lmList[4][:2], img)
                cursor = lmList[8][:2] #Index finger landmark
            except:
                # Default value when no hands are found.
                l = 100
            # Code runs for every hat added to the list.
            for hat in hatList:
                # Variables needed to be initialised before each iteration.
                imageSize = (200, 200)
                imageSizeScale = 700
                # Determining collision box for hat.
                w, h = hat.boundarySize
                # Location of the hat.
                cx, cy = hat.posCenter[0], hat.posCenter[1]
                # Checking if the index finger and thumb are close enough or not. (pinched)
                if l < 50:
                    # Determine if the index finger is in the collision box.
                    if (cx-w//2)<cursor[0]<(cx+w//2) and (cy-h//2)<cursor[1]<(cy+h//2):
                        # If the index finger is in the collision box the hat will update to the index finger position.
                        hat.update((cursor[0], cursor[1]))
                        # Since the hat is being picked up it is not attached to a face.
                        attached = False
                # When the index finger is not closed.
                else:
                    # Determine if the landmark at the top of the face mesh is in the hat collision box.
                    if (cx-w//2)<lmlist[10][0]<(cx+w//2) and (cy-h//2)<lmlist[10][1]<(cy+h//2):
                        # If the landmark is in the collision box the hat will update to the landmark position, making the person wear the hat.
                        hat.update((lmlist[10][0], lmlist[10][1]))
                        # Since it the person is wearing the hat, the hat is now attached, so it will follow the rotation and scale of the face mesh.
                        attached = True
                        # Checking  which hat is the one attached so that the only the one actually attached willbe affected by the rotation and scale of the face mesh.
                        currentAttached = hatList.index(hat)
                # Measuring the rotation of the face mesh.
                angle = checkAngle(lmlist[109], lmlist[338])
                # Checking if any hat is attached to a face mesh and which one is it.
                if attached == True and hatList.index(hat) == currentAttached:
                    # Calculating image scaling with respect to how close two points on the face mesh are, allowing for the hat to scale with distance from the camera.
                    imageSizeScale = int(hypot(lmlist[109][0] - lmlist[338][0], lmlist[109][1] - lmlist[338][1]))*12
                    # Resized hat image for when it is attached to face mesh.
                    hatSized = cv2.resize(hat.hat,(imageSizeScale,imageSizeScale))
                    # Finding the current size of the hat image.
                    imageSize = hatSized.shape[:2]
                    # Rotating the image with respect to the face mesh it is attached to using the angle initialised earlier.
                    hatRotatedKeepSize = rotateImage(hatSized, angle, scale=1,keepSize=True)
                # If the hat is not attached to face mesh.
                else:
                    # Hat resized to be smaller to reduce clutter in frame.
                    hatSized = cv2.resize(hat.hat,(200,200))
                    # No rotation applied to hat since it is not attached to face mesh.
                    hatRotatedKeepSize = hatSized
                # Rendering hats in the final frame.
                img2 = cvzone.overlayPNG(img, hatRotatedKeepSize, pos=(hat.posCenter[0]-imageSize[0]//2, hat.posCenter[1]-imageSize[1]//2))

                # This commented code is used for identifying the collision box.
                # cv2.circle(img2, (cx, cy), 20,  (255, 0, 255), cv2.FILLED)

                # This commented code is used for identifying the index of any point on the face mesh.
                # for i in lmlist:
                    # cv2.circle(img, i, 5,  (255, 0, 255), cv2.FILLED)
                    # cv2.putText(img, str(lmlist.index(i)), i, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # Display the frame in a window
    cv2.imshow("Camera", img2)
    # Wait for 1 millisecond between each frame
    cv2.waitKey(1)


    # Acknowledgements:
    # cvzone library at https://github.com/cvzone/cvzone/blob/master/README.md for the facial mesh and hand detection systems, as well as quality of life functions.
    # Pyzone for reference to how to overlay images on landmarks at https://www.youtube.com/watch?v=IJpTe-1cimE
