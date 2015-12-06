import sys
from PyQt4 import QtGui as gui
from PyQt4 import QtCore as core
from PyQt4 import uic
import cv2
import numpy as np
import math
from copy import deepcopy

#################################################### PARAMETERS #######################################################


thismodule = sys.modules[__name__] # for accessible global variables



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Video Source ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# local video source
file="/home/kyri/Videos/Tauben/FromWolfgang/Fassade_1.mp4"


# video input
thismodule.capture_source = 0 # file or 0 for camera
thismodule.capture = cv2.VideoCapture(thismodule.capture_source)


# video attributes
thismodule.width = int(thismodule.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
thismodule.height = int(thismodule.capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
thismodule.fourcc = thismodule.capture.get(cv2.cv.CV_CAP_PROP_FOURCC)
thismodule.actualFramerate = thismodule.capture.get(cv2.cv.CV_CAP_PROP_FPS)
if math.isnan(thismodule.actualFramerate) or thismodule.actualFramerate <= 0:    # sometimes getting FPS from a video file returns "NaN"
    print "framerate is NaN"
    thismodule.actualFramerate = 30              # most cases
thismodule.numframes=thismodule.capture.get(7)

# print some video information at the start of the application
print "\t"
print "\t Width: ", thismodule.width
print "\t Height: ", thismodule.height
print "\t FourCC: ", thismodule.fourcc
print "\t Framerate: ", thismodule.actualFramerate
print "\t Number of Frames: ", thismodule.numframes


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Background Subtractor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#  Backgroundsubtractor settings
#  Read for parameters:
#  http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html#backgroundsubtractormog
thismodule.history = 20
nGauss = 10
bgThresh = 0.2
noise = 0
thismodule.learningRate =  1 / float(thismodule.history)
bgs = cv2.BackgroundSubtractorMOG(thismodule.history, nGauss, bgThresh, noise)


# binary image
thismodule.showBinaryImage = False  # enable or disable the showing of the binary image output

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#  remove "salt" from binary image
kernelsize = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
iterations = 3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Draw Contours  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#  enable or disable the showing of contours on the output image
thismodule.showContours = False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Movement point system ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#  Scoring settings for movement detection
thismodule.movD = 10                                # defines a rectangle in which the next movement is recognised as the same
                                                    #  as the previous one

thismodule.pointVisibilityThreshold = 90            # movements are "collecting" points and when a movement supercedes this threshold
                                                    #  then it is a "legal" movement

thismodule.minusPoints = 2			                # decrease point when there is not enough movement

thismodule.plusPoints = 1                           # increase movement points

thismodule.plusPointsIfVisible = 1                  # add some bonus points

thismodule.maxPoints = 120                          # a upper limit of points a movement can have


# movement tracking
thismodule.movementList = []
thismodule.changes = []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bird Size ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# bird size
thismodule.defaultsize = 100     # number of pixels inside a contour (area)
thismodule.sizepercent = 20      # percentage to calculate a (min, max) range of area
                                 #  example of 100, 20: gives a range (80, 120)

# Area settings
thismodule.defMinArea = thismodule.defaultsize - (thismodule.defaultsize * (float(thismodule.sizepercent)/100)) # example of 100, 20: gives 80
thismodule.defMaxArea = thismodule.defaultsize + (thismodule.defaultsize * (float(thismodule.sizepercent)/100)) # example of 100, 20: gives 120

# Size Areas
thismodule.sizeregion = []            # stores custom set size areas (polygons). areas where not a custom area is defined uses the defaultsize
thismodule.currentsizeregion = [0]    # stores the edges of the currently defined size area polygon
 # initialize a black area (zeros) where later colored polygons will be drawn. the color of the polygon defines the size of the area; see size2color method
thismodule.sizemask = np.zeros((thismodule.height, thismodule.width, 3), dtype=np.uint8)
thismodule.prevmode = 0                    # need to be able to return to previous mode when finished defining size areas
thismodule.drawnewsizearea = False         # only be able to save a new size area if it has begun first

# Size Hint Rectangle
thismodule.rectsize = 0			   # a visual helper rectangle when defining new size areas to better estimate the needed size
thismodule.drawrect = False		   # only draw rectangle when changing values an hide it afterwards

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Circularity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Shape settings
# Circularity is a range (0,1) which defines how "round" an object is. 1 is a perfect circle
thismodule.minCircularity = 0.5 
thismodule.maxCircularity = 0.8


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Color ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Color settings
# Used to compare the objects' average color with this standard defined in the parameters (GUI) to see if it is a color of a real bird
thismodule.avCol = [0] * 3  # Average Color
thismodule.avCol[2] = 127   # red
thismodule.avCol[1] = 127   # green
thismodule.avCol[0] = 127   # blue
thismodule.inTol = 50       # intensityTolerance
thismodule.enableColorDetection = False # enable or disable color detection (can be used to boost performace)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Window Mode ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#window
# switches between the states of the application, with the following values:
thismodule.mode = 0                # 0 = only image, 1 = lines, 2 = poly,
                                   # 3 = size lines, 4 = show size areas,
                                   # 5 = laser lines, 6 = show laser areas

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Font ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

font = cv2.FONT_HERSHEY_SIMPLEX

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Resize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#resize
# the width and the height of the video are divided with this factor
thismodule.rx = 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Framerate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Framerate
# custom framerate set in the parameters
thismodule.newframerate = thismodule.actualFramerate
thismodule.everyNmili = 1000 / float(thismodule.newframerate)       # QTimer gets milliseconds parameter
                                                                    #  and applies filtering every N miliseconds

# used to calculate which frames are dropped when using custom framerate; see fillvalidframerate method
thismodule.validframes = []
for i in range(thismodule.actualFramerate):
    thismodule.validframes.append(1)
thismodule.framecounter = 0


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ROI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# ROI (polygons)
thismodule.region = []            # defining Regions of Interest with polygons
thismodule.currentregion = []     # currently defined polygon
thismodule.mask = np.zeros((thismodule.height, thismodule.width, 3), dtype=np.uint8)  # white polygons on black background
thismodule.drawnewpoly = False    # only be able to save a new ROI polygon if it has begun first

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Laser Areas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Laser Areas
thismodule.laserregion = []       # defining regions where the laser can shoot
thismodule.currentlaserregion = [] # currently defined polygon
thismodule.lasermask = np.zeros((thismodule.height, thismodule.width, 3), dtype=np.uint8) # size coded in color values; see size2color method
thismodule.drawnewlaser = False   # only be able to save a new laser area if it has begun first

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Logging
thismodule.logpause = 5                                    # log bird to shoot coordinates every n seconds in .csv file
thismodule.datetime = core.QDateTime()		               # used to log date and time in .csv file
thismodule.csvpath = "log/log1.csv"			               # log file destination
thismodule.topstring = "Date, Time, X, Y, Source\n"        # format of the line to be written
thismodule.lastLog = thismodule.datetime.currentDateTime() 
thismodule.csvFile = core.QFile(thismodule.csvpath)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bird to Shoot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

thismodule.birdToShoot = None          # only one bird can be shot at a time and here is stored which one (coordinates)
thismodule.maxMovePoints = -1          # when more than one birds are recognised at a time the movement point system
                                       #  is used to chose which bird (the one with the most points)


#######################################################################################################################

######################################################## WINDOWS ######################################################

# one class per window. 
# windows: main, parameters, about

# main window
class MyApp(gui.QMainWindow):

    frame = 1 # used to draw output image

    def __init__(self):
        gui.QMainWindow.__init__(self)

        # placeholders for "About" and "Parameters" windows
        self.about = None
        self.parameters = None

        self.ui = uic.loadUi('./ui/gui_menubar1.ui', self)                                                # self param to handle close button event!!
        self.ui.setWindowTitle("Taubenabwehr")
        self.ui.label.setGeometry(core.QRect(0,0,(thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx))) # resize window to video size
        self.ui.label.mousePressEvent = self.getPos                                                                   # mouse position
        self.ui.resize((thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx))                            # resize window to video size
        self.setup_actions()
        self.ui.show()
        self.setup_camera()

    # show message when trying to quit the application
    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = gui.QMessageBox.question(self, 'Message', quit_msg, gui.QMessageBox.Yes, gui.QMessageBox.No)
        if reply == gui.QMessageBox.Yes:
            self.destroyOpenWindows()
            event.accept()
        else:
            event.ignore()

    # close the other windows when quitting the application if they are open
    def destroyOpenWindows(self):
        if self.parameters is not None:                                         
            self.parameters.ui.destroy()
        if win.about is not None:
            self.about.ui.destroy()

    # get x,y coordinates of a mouse click and handle it depending on current mode
    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y()

        if thismodule.mode == 0:                                           # only image
            pass
        elif thismodule.mode == 1:                                         # lines
            thismodule.currentregion.append([x,y])
        elif thismodule.mode == 2:                                         # polygon
            pass
        elif thismodule.mode == 3:                                         # size lines
            thismodule.currentsizeregion.append([x,y])
        elif thismodule.mode == 4:                                         # show size areas
            pass
        elif thismodule.mode == 5:                                         # laser lines
            thismodule.currentlaserregion.append([x,y])
        elif thismodule.mode == 6:                                         # show laser areas
            pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ROI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # draw the lines that will later define the ROI polygon 
    def drawLines(self, img):

        # draw existing polygons
        nrofpolys = len(thismodule.region)
        if nrofpolys > 0:
            for i in range(nrofpolys):
                poly = thismodule.region[i]
                self.drawExistingPolyLines(img, poly)

        # draw the edges and the lines between them
        size = len(thismodule.currentregion)
        if size == 1:
            cv2.line(img, (thismodule.currentregion[0][0], thismodule.currentregion[0][1]), (thismodule.currentregion[0][0], thismodule.currentregion[0][1]), (1,5,255), 30)
        elif size > 1:
            for i in range(1, size):
                prev = thismodule.currentregion[i-1]
                curr = thismodule.currentregion[i]
                cv2.line(img, (prev[0], prev[1]), (prev[0], prev[1]), (1,5,255), 30)
                cv2.line(img, (prev[0], prev[1]), (curr[0], curr[1]), (255,5,255), 3)
            cv2.line(img, (thismodule.currentregion[size-1][0], thismodule.currentregion[size-1][1]), (thismodule.currentregion[size-1][0], thismodule.currentregion[size-1][1]), (1,5,255), 30)
            cv2.line(img, (thismodule.currentregion[size-2][0], thismodule.currentregion[size-2][1]), (thismodule.currentregion[size-1][0], thismodule.currentregion[size-1][1]), (255,5,255), 3)
        self.applyFiltering(img)

    # make everything black except for the polygon area (applying ROI)
    def drawPoly(self, img):
        if thismodule.drawnewpoly:
            poly = np.array((thismodule.currentregion), np.int32)
            cv2.fillPoly(thismodule.mask, [poly], (255,255,255))
            thismodule.drawnewpoly = False
        masked_img = cv2.bitwise_and(img, thismodule.mask)
        self.applyFiltering(masked_img)
        thismodule.currentregion = []

    # draw existing polygons
    def drawExistingPolyLines(self, img, poly):
        size = len(poly)
        for i in range(1, size):
            prev = poly[i-1]
            curr = poly[i]
            cv2.line(img, (prev[0], prev[1]), (curr[0], curr[1]), (0,0,255), 3)
        first = poly[0]
        last = poly[size-1]
        cv2.line(img, (first[0], first[1]), (last[0], last[1]), (0,0,255), 3)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Size Areas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # draw the size areas that have been definded with a size number in the center of each polygon
    def drawSetSizeAreas(self, img):
        nrofpolys = len(thismodule.sizeregion)
        if nrofpolys > 0:
            for i in range(nrofpolys):
                si = len(thismodule.sizeregion[i])
                poly = thismodule.sizeregion[i][1:si]
                self.drawExistingPolyLines(img, poly)
                closepoly = deepcopy(poly)
                closepoly.append(closepoly[0])
                cc = calculate_polygon_centroid(closepoly)
                cx = int(cc[0])
                cy = int(cc[1])
                areasize = str(thismodule.sizeregion[i][0])
                cv2.putText(img,areasize,(cx,cy), font, 1,(255,0,0),2,8)

    # draw the lines during the creation of a new size area
    def drawSizeAreaLines(self, img):
        nrofpolys = len(thismodule.sizeregion)
        if nrofpolys > 0:
            for i in range(nrofpolys):
                si = len(thismodule.sizeregion[i])
                poly = thismodule.sizeregion[i][1:si]
                self.drawExistingPolyLines(img, poly)
                closepoly = deepcopy(poly)
                closepoly.append(closepoly[0])
                cc = calculate_polygon_centroid(closepoly)
                cx = int(cc[0])
                cy = int(cc[1])
                areasize = str(thismodule.sizeregion[i][0])
                cv2.putText(img,areasize,(cx,cy), font, 1,(255,0,0),2,8)

        currsizereg = thismodule.currentsizeregion
        size = len(currsizereg)
        if size == 2:
            cv2.line(img, (currsizereg[1][0], currsizereg[1][1]), (currsizereg[1][0], currsizereg[1][1]), (1,5,255), 30)
        elif size > 2:
            for i in range(2, size):
                prev = currsizereg[i-1]
                curr = currsizereg[i]
                cv2.line(img, (prev[0], prev[1]), (prev[0], prev[1]), (1,5,255), 30)
                cv2.line(img, (prev[0], prev[1]), (curr[0], curr[1]), (255,5,255), 3)
            cv2.line(img, (currsizereg[size-1][0], currsizereg[size-1][1]), (currsizereg[size-1][0], currsizereg[size-1][1]), (1,5,255), 30)
            cv2.line(img, (currsizereg[size-2][0], currsizereg[size-2][1]), (currsizereg[size-1][0], currsizereg[size-1][1]), (255,5,255), 3)
        if thismodule.drawrect:
            uppercorner = 10
            val = int(round(math.sqrt(thismodule.rectsize)))
            cv2.rectangle(win.frame,(uppercorner,uppercorner),(uppercorner+val,uppercorner+val),(0,255,0),2)
        self.applyFiltering(img)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Laser Areas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # draw the lines that will later define the laser area polygon 
    def drawLaserAreaLines(self, img):

        # draw existing polygons
        nrofpolys = len(thismodule.laserregion)
        if nrofpolys > 0:
            for i in range(nrofpolys):
                poly = thismodule.laserregion[i]
                self.drawExistingPolyLines(img, poly)

        # draw the edges and the lines between them
        size = len(thismodule.currentlaserregion)
        if size == 1:
            cv2.line(img, (thismodule.currentlaserregion[0][0], thismodule.currentlaserregion[0][1]), (thismodule.currentlaserregion[0][0], thismodule.currentlaserregion[0][1]), (1,5,255), 30)
        elif size > 1:
            for i in range(1, size):
                prev = thismodule.currentlaserregion[i-1]
                curr = thismodule.currentlaserregion[i]
                cv2.line(img, (prev[0], prev[1]), (prev[0], prev[1]), (1,5,255), 30)
                cv2.line(img, (prev[0], prev[1]), (curr[0], curr[1]), (255,5,255), 3)
            cv2.line(img, (thismodule.currentlaserregion[size-1][0], thismodule.currentlaserregion[size-1][1]), (thismodule.currentlaserregion[size-1][0], thismodule.currentlaserregion[size-1][1]), (1,5,255), 30)
            cv2.line(img, (thismodule.currentlaserregion[size-2][0], thismodule.currentlaserregion[size-2][1]), (thismodule.currentlaserregion[size-1][0], thismodule.currentlaserregion[size-1][1]), (255,5,255), 3)
        self.applyFiltering(img)

    # show which laser areas have been defined
    def drawSetLaserAreas(self, img):
        if thismodule.drawnewlaser:
            poly = np.array((thismodule.currentlaserregion), np.int32)
            cv2.fillPoly(thismodule.lasermask, [poly], (255,255,255))
            thismodule.currentlaserregion = []
        thismodule.drawnewlaser = False
        nrofpolys = len(thismodule.laserregion)
        if nrofpolys > 0:
            for i in range(nrofpolys):
                si = len(thismodule.laserregion[i])
                poly = thismodule.laserregion[i]
                self.drawExistingPolyLines(img, poly)
        self.applyFiltering(img)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Movement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # awards points to continuous movement which are identified by occurrance in the same sector
    def increaseValidMovement(self, x, y, cnnt, circulBool, areaBool, colorBool):
        for i in range(len(thismodule.movementList)):
            a = thismodule.movementList[i]
            top, bot, right, left, cnt, c, cx, cy, cirB, arB, colB = a
            # check whether this movement in inside the bounding box of a preceding movement.
            if bot < y < top and left < x < right:
                if c > thismodule.pointVisibilityThreshold:
                    c += thismodule.plusPointsIfVisible
                else:
                    c += thismodule.plusPoints
                if c > thismodule.maxPoints:
                    c = thismodule.maxPoints
                thismodule.movementList[i] = y+thismodule.movD, y-thismodule.movD, x+thismodule.movD, x-thismodule.movD, cnnt, c, x, y, circulBool, areaBool, colorBool
                return i
        a = (y+thismodule.movD, y-thismodule.movD, x+thismodule.movD, x-thismodule.movD, cnnt, 1, x, y, circulBool, areaBool, colorBool)
        thismodule.movementList.append(a)
        return len(thismodule.movementList)-1

    # penalize if there was no movement in a specified sector
    def decreaseNoMovement(self, changes):
        copy = deepcopy(thismodule.movementList)
        toPop = []
        lCount = 0
        for i in range(len(copy)):
            if i not in changes:
                lCount += thismodule.plusPoints
                top, bot, right, left, cnt, c, cx, cy, cirB, arB, colB = copy[i]
                if c <= 0:
                    toPop.append(i)
                else:
                    c -= thismodule.minusPoints
                    thismodule.movementList[i] = top, bot, right, left, cnt, c, cx, cy, cirB, arB, colB
        toPop.sort()
        toPop.reverse()
        for i in toPop:
            thismodule.movementList.pop(i)
        return

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Filtering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # find the bird to shoot!
    def applyFiltering(self, img2):

        # binary image
        binary = bgs.apply(img2, learningRate=thismodule.learningRate) # background subtraction
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernelsize)  # remove "salt"

        if thismodule.showBinaryImage:
            cv2.imshow("binary", binary)

        # find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #
        thismodule.changes = []

        if len(contours) > 0:
            # iterate through all the moving objects
            for i in range(0, len(contours)):
                cnt = contours[i]    # get a single contour
                M = cv2.moments(cnt) # get image moments to determine the center of the contour
                if (M['m00'] != 0):
                    cx = int(M['m10']/M['m00']) # center-x
                    cy = int(M['m01']/M['m00']) # center-y
                    area = cv2.contourArea(cnt)
                    peri = cv2.arcLength(cnt,True) # perimeter of the contour
                    circul = 4 * math.pi * area / float(peri * peri)                # circularity; if the shape is close to a circle or not

                    # define the various "tests" a object has to pass in order to be recognised as a bird
                    circulBool = False
                    areaBool = False
                    colorBool = False

                    if thismodule.minCircularity < circul < thismodule.maxCircularity:
                        circulBool = True
                        birdSize = getSizeValue(cx, cy)

                        if birdSize == 0:
                            minArea = thismodule.defMinArea
                            maxArea = thismodule.defMaxArea
                        else:
                            minArea, maxArea = getMinMaxArea(birdSize)

                        if minArea < area < maxArea:
                            areaBool = True

                            if thismodule.enableColorDetection:
                                colorBool = self.checkAverageColor(cnt, img2)

                    if circulBool and areaBool:
                        i = self.increaseValidMovement(cx, cy, deepcopy(cnt), circulBool, areaBool, colorBool)
                        thismodule.changes.append(i)

                # draw contours around the object
                if thismodule.showContours:
                    cv2.drawContours(img2, [cnt], 0, (0, 255, 0), 1)

        self.decreaseNoMovement(thismodule.changes)

        for top, bot, right, left, cnt, c, cx, cy, cirB, arB, colB in thismodule.movementList:
            if c >= thismodule.pointVisibilityThreshold:

                # paint a red circle around a valid moving object
                cv2.circle(img2, (cx, cy), thismodule.movD, (255, 50, 220), 2)

                # check if bird is in a laser area (if allowed to be shot)
                inLaserArea = inShootingArea(cx, cy)
                allowedToShoot = cirB and arB and inLaserArea

                if thismodule.enableColorDetection and (not colB):
                    allowedToShoot = False

                if allowedToShoot:
                    # paint a green circle for valid use of laser
                    cv2.circle(img2, (cx, cy), thismodule.movD+3, (0, 255, 0), 2)

                    # when more than one birds are recognised at a time the movement point system
		            # is used to chose which bird (the one with the most points)
                    if c > thismodule.maxMovePoints:
                        thismodule.birdToShoot = [cx, cy]
                        thismodule.maxMovePoints = c

        self.shootBird()
	
        # draw image to frame
        self.frame = img2

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Color ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # compare the objects' average color with this standard defined in the parameters (GUI) to see if it is a color of a real bird
    def checkAverageColor(self, cnt, img2):
        # get the most outer Points of the contour
        l = tuple(cnt[cnt[:, :, 0].argmin()][0])
        r = tuple(cnt[cnt[:, :, 0].argmax()][0])
        t = tuple(cnt[cnt[:, :, 1].argmin()][0])
        b = tuple(cnt[cnt[:, :, 1].argmax()][0])

        aPix = [0] * 3
        count = 0
        p1 = (l[0]-10, t[1]-10)
        p2 = (r[0]+10, b[1]+10)
        if p2[0]-p1[0] > 0 and p2[1]-p1[1] > 0:
            for yi in range(p1[0], p2[0]):
                for xi in range(p1[1], p2[1]):
                    h, w, depth = img2.shape
                    if xi < w/2 and yi < h/2:
                        if cv2.pointPolygonTest(cnt, (yi, xi), False) != -1:
                            pix = img2[xi, yi]  # get current pixel value
                            aPix[0] += pix[0]  # blue
                            aPix[1] += pix[1]  # green
                            aPix[2] += pix[2]  # red
                            count += 1

        if count > 0:
            aPix[0] /= count
            aPix[1] /= count
            aPix[2] /= count

        #  Return true if all the intensities are within the accepted range
        if aPix[0] - thismodule.inTol <= thismodule.avCol[1] <= aPix[1] + thismodule.inTol \
                and aPix[1] - thismodule.inTol <= thismodule.avCol[1] <= aPix[1] + thismodule.inTol\
                and aPix[1] - thismodule.inTol <= thismodule.avCol[1] <= aPix[1] + thismodule.  inTol:
            return True
        else:
            return False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Actions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # what happens when we decide to shoot a bird; can be used to forward the bird's coordinates to the laser machine
    def shootBird(self):
        if thismodule.birdToShoot is not None:
            if CanLog():
                newLogEntry()

        thismodule.birdToShoot = None
        thismodule.maxMovePoints = -1

    # binding each menu item to an action (what happens when clicked); see Action Functions
    def setup_actions(self):
        self.ui.actionQuit.triggered.connect(QuitAction)
        self.ui.actionBegin_Polygon.triggered.connect(BeginPolygonAction)
        self.ui.actionDraw_Polygon.triggered.connect(DrawPolygonAction)
        self.ui.actionReset.triggered.connect(ResetAction)
        self.ui.actionBegin_SArea.triggered.connect(BeginSAreaAction)
        self.ui.actionEnd_SArea.triggered.connect(EndSAreaAction)
        self.ui.actionExit_SAreas.triggered.connect(ExitSAreasAction)
        self.ui.actionDraw_SArea.triggered.connect(ViewAreaAction)
        self.ui.actionReset_SAreas.triggered.connect(ResetSAreasAction)
        self.ui.actionAbout.triggered.connect(AboutAction)
        self.ui.actionParameters.triggered.connect(ParametersAction)
        self.ui.actionSource.triggered.connect(VideoSourceAction)
        self.ui.actionBegin_Laser.triggered.connect(BeginLaserAreaAction)
        self.ui.actionDraw_LaserArea.triggered.connect(DrawLaserAreaAction)
        self.ui.actionExit_LaserArea.triggered.connect(ExitSAreasAction) # same action
        self.ui.actionRemove_LaserAreas.triggered.connect(ResetLaserAreasAction)
        self.ui.actionRestart.triggered.connect(RestartVideoAction)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output Image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    # setup a timer that displays the video stream every n milliseconds
    def setup_camera(self):
        self.timer = core.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        thismodule.everyNmili = 1000 / float(thismodule.newframerate)
        self.timer.start(thismodule.everyNmili)

    # show the output image
    def display_video_stream(self):
        if thismodule.validframes[thismodule.framecounter] == 0:
            thismodule.capture.grab() # drop frame, more efficient than read()
            incrementCounter()
        else:
            ok, self.frame = thismodule.capture.read()
            incrementCounter()

            if ok:
                self.frame = cv2.cvtColor(self.frame, cv2.cv.CV_BGR2RGB)
                self.frame = cv2.flip(self.frame, 1)
                self.frame = cv2.resize(self.frame, (0, 0), fx=(1/float(thismodule.rx)), fy=(1/float(thismodule.rx)))

                if thismodule.mode == 0:                                           # only image
                    self.applyFiltering(self.frame)
                elif thismodule.mode == 1:                                         # lines
                    self.drawLines(self.frame)
                elif thismodule.mode == 2:                                         # polygon
                    self.drawPoly(self.frame)
                elif thismodule.mode == 3:                                         # size area lines
                    self.drawSizeAreaLines(self.frame)
                elif thismodule.mode == 4:                                         # show size areas
                    self.drawSetSizeAreas(self.frame)
                elif thismodule.mode == 5:                                          # laser area lines
                    self.drawLaserAreaLines(self.frame)
                elif thismodule.mode == 6:                                          # show laser areas
                    self.drawSetLaserAreas(self.frame)

                # draw image on the central label
                image = gui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0], gui.QImage.Format_RGB888)
                self.ui.label.setPixmap(gui.QPixmap.fromImage(image))
            else:
                print "no more frames"

# about window
class AboutDialog(gui.QWidget):
    def __init__(self):
        gui.QWidget.__init__(self)
        self.ui = uic.loadUi('./ui/about_dialog.ui')
        self.ui.setWindowTitle("About")

# parameters window
class ParametersWindow(gui.QWidget):
    def __init__(self):
        gui.QWidget.__init__(self)
        self.ui = uic.loadUi('./ui/parameters.ui')
        self.ui.setWindowTitle("Parameters")
        self.configureObjects()

    # set input validators and show the current values
    def configureObjects(self):
        self.ui.lineEdit_framerate.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_framerate.setText(core.QString.number(thismodule.newframerate))

        self.ui.lineEdit_resize.setValidator(gui.QDoubleValidator(self))
        self.ui.lineEdit_resize.setText(core.QString.number(thismodule.rx))

        self.ui.lineEdit_mincircularity.setValidator(gui.QDoubleValidator(self))
        self.ui.lineEdit_mincircularity.setText(core.QString.number(thismodule.minCircularity))

        self.ui.lineEdit_maxcircularity.setValidator(gui.QDoubleValidator(self))
        self.ui.lineEdit_maxcircularity.setText(core.QString.number(thismodule.maxCircularity))

        self.ui.lineEdit_defsize.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_defsize.setText(core.QString.number(thismodule.defaultsize))

        self.ui.lineEdit_tolerance.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_tolerance.setText(core.QString.number(thismodule.sizepercent))

        self.ui.lineEdit_logpause.setValidator(gui.QDoubleValidator(self))
        self.ui.lineEdit_logpause.setText(core.QString.number(thismodule.logpause))

        self.ui.lineEdit_pointthresh.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_pointthresh.setText(core.QString.number(thismodule.pointVisibilityThreshold))

        self.ui.lineEdit_movD.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_movD.setText(core.QString.number(thismodule.movD))

        self.ui.lineEdit_pluspoints.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_pluspoints.setText(core.QString.number(thismodule.plusPoints))

        self.ui.lineEdit_minuspoints.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_minuspoints.setText(core.QString.number(thismodule.minusPoints))

        self.ui.lineEdit_maxpoints.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_maxpoints.setText(core.QString.number(thismodule.maxPoints))

        self.ui.lineEdit_history.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_history.setText(core.QString.number(thismodule.history))

        self.ui.lineEdit_R.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_R.setText(core.QString.number(thismodule.avCol[2]))

        self.ui.lineEdit_G.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_G.setText(core.QString.number(thismodule.avCol[1]))

        self.ui.lineEdit_B.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_B.setText(core.QString.number(thismodule.avCol[0]))

        self.ui.lineEdit_tolerancergb.setValidator(gui.QIntValidator(self))
        self.ui.lineEdit_tolerancergb.setText(core.QString.number(thismodule.inTol))

        enableColorState = core.Qt.Unchecked
        if thismodule.enableColorDetection:
            enableColorState = core.Qt.Checked
        self.ui.checkBox_enablecolor.setCheckState(enableColorState)

        showBinaryState = core.Qt.Unchecked
        if thismodule.showBinaryImage:
            showBinaryState = core.Qt.Checked
        self.ui.checkBox_showbinary.setCheckState(showBinaryState)

        showContoursState = core.Qt.Unchecked
        if thismodule.showContours:
            showContoursState = core.Qt.Checked
        self.ui.checkBox_showcontours.setCheckState(showContoursState)

        self.ui.pushButton_save.clicked.connect(SaveParametersAction)
        self.ui.pushButton_cancel.clicked.connect(CancelParametersAction)


#######################################################################################################################

################################################ ACTION FUNCTIONS #####################################################

# What happens when you click a menu item

def QuitAction():
    sys.exit(0)

def BeginPolygonAction():
    setMode(1)                  # ROI lines
    thismodule.drawnewpoly = True

def DrawPolygonAction():
    if thismodule.drawnewpoly:
        if len(thismodule.currentregion)>2 :
            tmpregion = deepcopy(thismodule.currentregion)
            thismodule.region.append(tmpregion)
            if len(thismodule.region)>0 :
                setMode(2)          # show ROI polygons
        else:
            gui.QMessageBox.warning(win, "Warning", "Needs at least 3 polygon points first")
    elif len(thismodule.region)>0:
        setMode(2)          # show ROI polygons


def DrawLaserAreaAction():
    if thismodule.drawnewlaser:
        if len(thismodule.currentlaserregion)>2 :
            tmpregion = deepcopy(thismodule.currentlaserregion)
            thismodule.laserregion.append(tmpregion)

            if len(thismodule.laserregion)>0 :
                setMode(6)          # show Laser Areas
        else:
            gui.QMessageBox.warning(win, "Warning", "Needs at least 3 polygon points first")
    elif len(thismodule.laserregion)>0:
        setMode(6)

def ResetAction():
    resetMode()                 # image only
    resetROI()

def BeginSAreaAction():
    thismodule.drawnewsizearea = True
    thismodule.prevmode = deepcopy(thismodule.mode)
    setMode(3)                  # size area lines

def BeginLaserAreaAction():
    thismodule.drawnewlaser = True
    thismodule.prevmode = deepcopy(thismodule.mode)
    setMode(5)                  # laser area lines

def EndSAreaAction():
    if thismodule.drawnewsizearea:
        areasize = len(thismodule.currentsizeregion)
        #print "areasize: ", areasize
        if areasize>3 :
            inputDia = gui.QInputDialog(win)
            inputDia.setInputMode(gui.QInputDialog.IntInput)
            inputDia.setIntValue(0)
            inputDia.setIntRange(0, 65535)
            inputDia.setIntStep(1)
            inputDia.setLabelText("Size [0, 65535]:")
            inputDia.intValueChanged.connect(IntValueChangeAction)
            inputDia.intValueSelected.connect(IntValueSelectedAction)
            inputDia.open()
            # continues in IntValueSelectedAction(val) method
        else:
            gui.QMessageBox.warning(win, "Warning", "Needs at least 3 polygon points first")

def IntValueChangeAction(val):
    thismodule.drawrect = True
    thismodule.rectsize = val

def IntValueSelectedAction(val):
    thismodule.drawrect = False
    thismodule.rectsize = 0
    thismodule.currentsizeregion[0] = val
    areasize = len(thismodule.currentsizeregion)
    #if areasize>1 :
    tmpregion = deepcopy(thismodule.currentsizeregion)
    color = size2Color(tmpregion[0])
    poly = np.array((tmpregion[1:areasize]), np.int32)
    cv2.fillPoly(thismodule.sizemask, [poly], color)
    thismodule.sizeregion.append(tmpregion)
    thismodule.currentsizeregion = [0]
    thismodule.mode = deepcopy(thismodule.prevmode)
    thismodule.drawnewsizearea = False


def VideoSourceAction():
    t, ok = gui.QInputDialog.getText(win, "QInputDialog.getText()", "/path/to/video.mp4 or 0,1,.. for camera(s)")
    t = str(t)
    if len(t) == 1 and t.isdigit():
        t = int(t)
    elif len(t) == 0:
        ok = False

    if ok:
        testcapture = cv2.VideoCapture(t)
        if testcapture.isOpened():
            testcapture = None
            setVideoSource(t)
        else:
            print "video source not ok"
    else:
        print "video source not ok"


def ViewAreaAction():
    setMode(4)          # show size areas


def ExitSAreasAction():
    if len(thismodule.region)>0:
        setMode(2)                  # draw ROI polygons
    else:
        setMode(0)                  # image only


def AboutAction():
    win.about = AboutDialog()
    win.about.ui.show()


def ResetSAreasAction():
    if len(thismodule.region)>0:
        setMode(2)                  # draw ROI polygons
    else:
        setMode(0)                  # image only
    resetSizeAreas()

def ResetLaserAreasAction():
    if len(thismodule.region)>0:
        setMode(2)                  # draw ROI polygons
    else:
        setMode(0)                  # image only
    resetLaserAreas()

def ParametersAction():
    if win.parameters is None:
        win.parameters = ParametersWindow()
    win.parameters.ui.show()

# update and save values given in parameters window
def SaveParametersAction():
    i =  win.parameters.ui.lineEdit_framerate.text()
    updateFramerate(i)

    res = win.parameters.ui.lineEdit_resize.text()
    if thismodule.rx != float(res):
        resizeVideo(res)

    minc = win.parameters.ui.lineEdit_mincircularity.text()
    minc = float(minc)
    thismodule.minCircularity = minc

    maxc = win.parameters.ui.lineEdit_maxcircularity.text()
    maxc = float(maxc)
    thismodule.maxCircularity = maxc

    to = win.parameters.ui.lineEdit_tolerance.text()
    to = int(to)
    thismodule.sizepercent = to

    ds = win.parameters.ui.lineEdit_defsize.text()
    ds = int(ds)
    thismodule.defaultsize = ds
    thismodule.defMinArea = thismodule.defaultsize - (thismodule.defaultsize * (100/float(thismodule.sizepercent)))
    thismodule.defMaxArea = thismodule.defaultsize + (thismodule.defaultsize * (100/float(thismodule.sizepercent)))

    lp = win.parameters.ui.lineEdit_logpause.text()
    lp = float(lp)
    thismodule.logpause = lp

    thismodule.enableColorDetection = win.parameters.ui.checkBox_enablecolor.isChecked()

    thismodule.showBinaryImage = win.parameters.ui.checkBox_showbinary.isChecked()

    thismodule.showContours = win.parameters.ui.checkBox_showcontours.isChecked()

    pnt = win.parameters.ui.lineEdit_pointthresh.text()
    pnt = int(pnt)
    thismodule.pointVisibilityThreshold = pnt

    mvd = win.parameters.ui.lineEdit_movD.text()
    mvd = int(mvd)
    thismodule.movD = mvd

    plspnts = win.parameters.ui.lineEdit_pluspoints.text()
    plspnts = int(plspnts)
    thismodule.plusPoints = plspnts

    mnspnts = win.parameters.ui.lineEdit_minuspoints.text()
    mnspnts = int(mnspnts)
    thismodule.minusPoints = mnspnts

    mxpnts = win.parameters.ui.lineEdit_maxpoints.text()
    mxpnts = int(mxpnts)
    thismodule.maxPoints = mxpnts

    hstry = win.parameters.ui.lineEdit_history.text()
    hstry = int(hstry)
    thismodule.history = hstry
    thismodule.learningRate = float(1) / thismodule.history

    r = win.parameters.ui.lineEdit_R.text()
    r = int(r)
    thismodule.avCol[2] = r

    g = win.parameters.ui.lineEdit_G.text()
    g = int(g)
    thismodule.avCol[1] = g

    b = win.parameters.ui.lineEdit_B.text()
    b = int(b)
    thismodule.avCol[0] = b

    tolrgb = win.parameters.ui.lineEdit_tolerancergb.text()
    tolrgb = int(tolrgb)
    thismodule.inTol = tolrgb


def CancelParametersAction():
    win.parameters.ui.hide()

def RestartVideoAction(): # if source is file
    setVideoSource(thismodule.capture_source)

#######################################################################################################################

#################################################### AUXILIARY FUNCTIONS ##############################################

# only log a bird's position if at least <logpause> seconds have passed since last log
def CanLog():
    currDT = thismodule.datetime.currentDateTime()
    msecDiff = abs(currDT.msecsTo(thismodule.lastLog))
    can =  msecDiff > (thismodule.logpause * 1000)
    if can:
        thismodule.lastLog = thismodule.datetime.currentDateTime()
    return can

def newLogEntry():
    print "shoot bird at position: ", thismodule.birdToShoot
    thismodule.csvFile.open(core.QIODevice.WriteOnly | core.QIODevice.Text | core.QIODevice.Append)
    if thismodule.csvFile.size() ==0:
        print "file is empty"
        thismodule.csvFile.write(thismodule.topstring)

    cDate = str(thismodule.datetime.date().currentDate().toString())
    cTime = str(thismodule.datetime.time().currentTime().toString())
    X = str(thismodule.birdToShoot[0])
    Y = str(thismodule.birdToShoot[1])
    source = str(thismodule.capture_source)
    strData =  cDate + "," + cTime + "," + X + "," + Y + "," + source + "\n"
    thismodule.csvFile.write(strData)
    thismodule.csvFile.close()

# custom framerate
def updateFramerate(i):
    thismodule.newframerate = int(i)
    fillValidFramerates()

def setVideoSource(vsource):
    thismodule.capture_source = vsource # 0 for camera
    thismodule.capture = cv2.VideoCapture(thismodule.capture_source)
    resetVideoAttributes()
    resetROI()
    resetSizeAreas()
    resetLaserAreas()
    resetMode()
    win.ui.label.setGeometry(core.QRect(0,0,(thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx))) # resize window to video size
    win.ui.resize((thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx))


def resetVideoAttributes():
    thismodule.width = int(thismodule.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    thismodule.height = int(thismodule.capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    thismodule.fourcc = thismodule.capture.get(cv2.cv.CV_CAP_PROP_FOURCC)
    thismodule.actualFramerate = thismodule.capture.get(cv2.cv.CV_CAP_PROP_FPS)
    if math.isnan(thismodule.actualFramerate) or thismodule.actualFramerate <= 0:
        print "framerate is NaN"
        thismodule.actualFramerate = 30
    thismodule.numframes=thismodule.capture.get(7)

    print "\t"
    print "\t Width: ", thismodule.width
    print "\t Height: ", thismodule.height
    print "\t FourCC: ", thismodule.fourcc
    print "\t Framerate: ", thismodule.actualFramerate
    print "\t Number of Frames: ", thismodule.numframes


def resetROI():
    thismodule.region = []
    thismodule.currentregion = []
    thismodule.mask = np.zeros(((thismodule.height/thismodule.rx), (thismodule.width/thismodule.rx), 3), dtype=np.uint8)
    thismodule.drawnewpoly = False

def resetSizeAreas():
    thismodule.sizeregion = []                                                                 
    thismodule.currentsizeregion = [0]
    thismodule.sizemask = np.zeros(((thismodule.height/thismodule.rx), (thismodule.width/thismodule.rx), 3), dtype=np.uint8) 
    thismodule.prevmode = 0    


def resetLaserAreas():
    thismodule.laserregion = []
    thismodule.currentlaserregion = []
    thismodule.lasermask = np.zeros(((thismodule.height/thismodule.rx), (thismodule.width/thismodule.rx), 3), dtype=np.uint8)
    thismodule.drawnewlaser = False

def setMode(m):
    thismodule.mode = m

def resetMode():
    setMode(0)

def resizeVideo(resx):
    thismodule.rx = float(resx)
    resetMode()
    resetROI()
    resetSizeAreas()
    resetLaserAreas()
    win.ui.label.setGeometry(core.QRect(0,0,(thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx)))  # resize window to video size
    win.ui.resize((thismodule.width/thismodule.rx),(thismodule.height/thismodule.rx))

# decode polygon color to size of a sizearea
def size2Color(x):          # [0, 65535]
    r = x / 256
    g = x % 256
    b = 0
    return (r,g,b)

# encode size of a sizearea to polygon color
def color2size((r,g,b)):
    s = (r*256) + g
    return s

# get the expected size of a bird at position x,y
def getSizeValue(x,y):
    r = thismodule.sizemask[y][x][0]
    g = thismodule.sizemask[y][x][1]
    b = 0
    s = color2size((r,g,b))
    return s

# check if bird is in a laser area
def inShootingArea(x,y):
    r = thismodule.lasermask[y][x][0]
    g = thismodule.lasermask[y][x][1]
    b = thismodule.lasermask[y][x][2]
    return (r == 255) and (g == 255) and (b == 255)

# create a range (min, max) of expected size area
def getMinMaxArea(x):
    factor = float(thismodule.sizepercent)/100
    min = x - (x*factor)
    max = x + (x*factor)
    return (min,max)

# in custom framerate define which frames are dropped
def fillValidFramerates():
    initValidFrames(1)
    if thismodule.newframerate < thismodule.actualFramerate:
        nrFramesToDrop = thismodule.actualFramerate - thismodule.newframerate
        dropFactor = thismodule.actualFramerate / float(nrFramesToDrop)
        for i in range(1, (nrFramesToDrop+1)):
                currFrame = i * dropFactor
                exactFrame = int(round(currFrame)) -1
                thismodule.validframes[exactFrame] = 0

# helper function to init valid frames array
def initValidFrames(k):
    thismodule.validframes = []
    for i in range(thismodule.actualFramerate):
        thismodule.validframes.append(k)

# iterate through valid frames array
def incrementCounter():
    thismodule.framecounter = thismodule.framecounter + 1
    if thismodule.framecounter > thismodule.actualFramerate-1:
        thismodule.framecounter = 0

# copied code from this link:
# https://raw.githubusercontent.com/pwcazenave/pml-git/master/python/centroids.py
def calculate_polygon_area(polygon, signed=False):
    """Calculate the signed area of non-self-intersecting polygon

    Input
        polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
        signed: Optional flag deciding whether returned area retains its sign:
                If points are ordered counter clockwise, the signed area
                will be positive.
                If points are ordered clockwise, it will be negative
                Default is False which means that the area is always positive.
    Output
        area: Area of polygon (subject to the value of argument signed)
    """

    # Make sure it is numeric
    P = np.array(polygon)

    # Check input
    msg = ('Polygon is assumed to consist of coordinate pairs. '
           'I got second dimension %i instead of 2' % P.shape[1])
    assert P.shape[1] == 2, msg

    msg = ('Polygon is assumed to be closed. '
           'However first and last coordinates are different: '
           '(%f, %f) and (%f, %f)' % (P[0, 0], P[0, 1], P[-1, 0], P[-1, 1]))
    assert np.allclose(P[0, :], P[-1, :]), msg

    # Extract x and y coordinates
    x = P[:, 0]
    y = P[:, 1]

    # Area calculation
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]
    A = np.sum(a - b) / 2.

    # Return signed or unsigned area
    if signed:
        return A
    else:
        return abs(A)

# copied code from this link:
# https://raw.githubusercontent.com/pwcazenave/pml-git/master/python/centroids.py
def calculate_polygon_centroid(polygon):
    """Calculate the centroid of non-self-intersecting polygon

    Input
        polygon: Numeric array of points (longitude, latitude). It is assumed
                 to be closed, i.e. first and last points are identical
    Output
        Numeric (1 x 2) array of points representing the centroid
    """

    # Make sure it is numeric
    P = np.array(polygon)

    # Get area - needed to compute centroid
    A = calculate_polygon_area(P, signed=True)

    # Extract x and y coordinates
    x = P[:, 0]
    y = P[:, 1]

    # Exercise: Compute C as shown in http://paulbourke.net/geometry/polyarea
    a = x[:-1] * y[1:]
    b = y[:-1] * x[1:]

    cx = x[:-1] + x[1:]
    cy = y[:-1] + y[1:]

    Cx = np.sum(cx * (a - b)) / (6. * A)
    Cy = np.sum(cy * (a - b)) / (6. * A)

    # Create Nx2 array and return
    C = np.array([Cx, Cy])
    return C

#######################################################################################################################

# application starter
if __name__ == "__main__":
    app = gui.QApplication(sys.argv)
    app.setWindowIcon(gui.QIcon('icon/christmas-tree-icon.png'))
    win = MyApp()
    sys.exit(app.exec_())