# total 
# sum of all total euclidian distances from point / number of data points
import os
import csv
import cv2
from shutil import copyfile
import numpy as np

directory = "./csvtest/"
totalError = 0
totalAcc = 0
i = 0
for dirpath,_,filename in os.walk(directory):
    for f in filename:
    	with open(os.path.join(directory, f)) as f:
    		readCSV = csv.reader(f, delimiter=',')
	    	for row in readCSV:
	    		i += 1
	    		frameFilename = row[0]
	    		frameTimestamp = row[1]
	    		tobiiLeftEyeGazeX = float( row[2] )
	    		tobiiLeftEyeGazeY = float( row[3] )
	    		tobiiRightEyeGazeX = float( row[4] )
	    		tobiiRightEyeGazeY = float( row[5] )
	    		webgazerX = float( row[6] )
	    		webgazerY = float( row[7] )
	    		clmTracker = row[8:len(row)-1]
	    		clmTracker = [float(i) for i in clmTracker]
	    		clmTrackerInt = [int(i) for i in clmTracker]
	    		tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
	    		tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

	    		webgazerPoint = np.array([webgazerX, webgazerY])
	    		tobiiPoint = np.array([tobiiEyeGazeX, tobiiEyeGazeY])
	    		error = abs(np.linalg.norm(webgazerPoint-tobiiPoint))
	    		acc = 1 - error
	    		totalError += error
	    		totalAcc += acc

avgError = totalError / i
avgAcc = totalAcc / i
# print ("avgError: %s" %avgError)
print ("WebGazer Average Accuracy: %s " %avgAcc)

        	
	    	       