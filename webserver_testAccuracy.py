# This file is a work in progress!
# As in, James is working on it right now.
# Dev = Deploy Env 4 Life.
#


#!/usr/bin/env python

import logging
import os
import sys
import time
import tornado.httpserver
import tornado.ioloop
import tornado.log
import tornado.web
import base64
from binascii import a2b_base64
import re
import json, csv
from decimal import Decimal
import math
import glob
from datetime import datetime

dt = datetime.now()
dtStr = dt.strftime('%Y%m%d_%H%M%S') + "_AccuracyEvaluation"
os.makedirs( dtStr )

participantDirPrev = ""
videoFilePrev = ""
tobiiListPos = 1
tobiiList = []

screenWidthPixels = -1
screenHeightPixels = -1
wgWindowX = -1
wgWindowY = -1
wgWindowInnerWidth = -1
wgWindowInnerHeight = -1
wgWindowOuterWidth = -1
wgWindowOuterHeight = -1

# WARNING: MAGIC NUMBER
# In pixels
# Looks like Chrome doesn't change the height of this bar as desktop scaling changes, so we're in luck
chromeDownloadBarHeight = 52

# Averages for errors per video
tobiiWGDiffPerVideo = 0
tobiiWGNumFrames = 0
# Tobii/WebGazer average difference per video
diffPerVideoCSV = dtStr + '/tobiiWebGazerAverageDiffPerVideo.csv'

class TobiiData:
    timestamp = 0
    rightEyeValid = -math.inf
    leftEyeValid = -math.inf
    rightScreenGazeX = -math.inf
    rightScreenGazeY = -math.inf
    leftScreenGazeX = -math.inf
    leftScreenGazeY = -math.inf

    def __init__(self, timestamp, rev, lev, rsgx, rsgy, lsgx, lsgy ):
        self.timestamp = timestamp
        self.rightEyeValid = rev
        self.leftEyeValid = lev
        self.rightScreenGazeX = rsgx
        self.rightScreenGazeY = rsgy
        self.leftScreenGazeX = lsgx
        self.leftScreenGazeY = lsgy

    def __str__(self):
        return "[TobiiData] Epoch: " + str(self.epoch) + "  RightEyeValid: " + str(self.rightEyeValid) + "  REX: " + str(self.rightScreenGazeX) + "  REY: " + str(self.rightScreenGazeY)

class WebgazerHandler(tornado.web.RequestHandler):

    def get(self):
        global tobiiListPos, participantDirPrev, videoFilePrev
        global screenWidthPixels, screenHeightPixels
        global tobiiWGDiffPerVideo, tobiiWGNumFrames, diffPerVideoCSV, dtStr
        global wgWindowX, wgWindowY, wgWindowInnerWidth, wgWindowInnerHeight, wgWindowOuterWidth, wgWindowOuterHeight

        timestamp = self.get_argument('epoch')
        timestampInt = int( round( float( timestamp ) ) )
        video = self.get_argument('video')
        # WebGazer prediction from browser
        wgPredX = float(self.get_argument('predX'))
        wgPredY = float(self.get_argument('predY'))

        # Split the video name into its pieces
        ind = video.find('/')
        videoFolder = video[0:ind]
        videoFile = video[ind+1:len(video)]
        ind = videoFile.rfind('.')
        videoName = videoFile[0:ind]
        participantStartTimestamp = videoFile[0:videoFile.find('_')]
        videoExt = videoFile[ind+1:len(videoFile)]
        # Participant characteristics (technical, e.g., screen resolution)
        pctFile = "participant_characteristics_technical.csv"
        # Tobii JSON
        tobiiJSON = videoFolder + '/' + videoFolder + ".txt"
        # WebGazer event log
        wgEventLog = videoFolder + "/" + participantStartTimestamp + ".txt"
        # Target dir for output
        outDir = dtStr + '/' + videoFolder + "/" + videoName + "_eval"
        # Target frame name
        fname =  outDir + "/" + str(timestampInt) + ".png"
        # Tobii/WebGazer difference per frame
        diffCSV = outDir + '/' + '/tobiiWebGazerDifference.csv'

        ###############################################################################################
        # Check if we have the right Tobii data loaded
        # Directory to process
        if participantDirPrev != videoFolder:
            # If not, load it.
            print( "Loading Tobii data for participant " + videoFolder )
            participantDirPrev = videoFolder
    
            # Load participant characteristics as technical parts
            with open( pctFile ) as f:
                readCSV = csv.reader(f, delimiter=',')
                for row in readCSV:
                    if row[0] == videoFolder:
                        screenWidthPixels = int(row[4])
                        screenHeightPixels = int(row[5])
                        break

            # Load WebGazer browser window parameters
            with open( wgEventLog ) as f:
                for line in f:
                    l = json.loads(line)
                    if l.get('windowX') != None:
                        wgWindowX = int(l['windowX'])
                        wgWindowY = int(l['windowY'])
                        wgWindowInnerWidth = int(l['windowInnerWidth'])
                        wgWindowInnerHeight = int(l['windowInnerHeight'])
                        wgWindowOuterWidth = int(l['windowOuterWidth'])
                        wgWindowOuterHeight = int(l['windowOuterHeight'])
                        break

            # Read in JSON output from Tobii
            # Each line is a JSON object, so let's read the file line by line
            with open( tobiiJSON, 'r' ) as f:
                for line in f:
                    l = json.loads(line, parse_float=Decimal)
                    
                    rsgx = l['right_gaze_point_on_display_area'][0]
                    rsgy = l['right_gaze_point_on_display_area'][1]
                    lsgx = l['left_gaze_point_on_display_area'][0]
                    lsgy = l['left_gaze_point_on_display_area'][1]
                    timestamp = round( l['system_time_stamp'] / 1000 )
                    td = TobiiData( timestamp, l['right_pupil_validity'], l['left_pupil_validity'], rsgx, rsgy, lsgx, lsgy )
                    tobiiList.append( td )

        if videoFilePrev != videoFile:
            # Make an output directory for the video frames
            if not os.path.isdir(outDir):
                os.makedirs(outDir)

            if videoFilePrev != "":
                # Compute average Tobii/WebGazer difference per video
                avgDiff = tobiiWGDiffPerVideo / tobiiWGNumFrames
                with open( diffPerVideoCSV, 'a') as f:
                    f.write( videoFolder + "/" + videoFilePrev + "," + str(avgDiff) + "\n")

            # Reset the running average of the error per video
            tobiiWGDiffPerVideo = 0
            tobiiWGNumFrames = 0
            
            videoFilePrev = videoFile
            print( "New video timestamp: " + str(timestampInt) + "  Video:  " + video )
            # Reset the timestamp running index
            # The video files are out of chronological order by file name.
            tobiiListPos = 1

            

        ###########################################################################################################
        # Next, find the closest Tobii timestamp to our current video timestamp
        #
        # As time only goes forwards, tobiiListPos is a counter which persists over GET requests.
        # The videos arrive in non-chronological order, however, so we have to reset tobiiListPos on each new video
        while timestampInt - tobiiList[tobiiListPos].timestamp > 0 and tobiiListPos < len(tobiiList)-1:
            tobiiListPos = tobiiListPos + 1

        diffCurr = timestampInt - tobiiList[tobiiListPos].timestamp
        diffPrev = timestampInt - tobiiList[tobiiListPos-1].timestamp

        # Pick the one which is closest in time
        if abs(diffCurr) < abs(diffPrev):
            td = tobiiList[tobiiListPos]
        else:
            td = tobiiList[tobiiListPos-1]

        #print( abs( td.timestamp - timestampInt ) )

        # Check that the tobii timestamp is within 33 milliseconds (~1 frame) of the webgazer timestamp
        # If not, it's not worth comparing them, so do nothing
        if abs( td.timestamp - timestampInt ) > 33:
            return

        # Some output if you want to compare timestamps...
        # Not that just because we've find a close match in time, doesn't mean that the 
        # timestamp coming from the browser is actually correct...
        #print( "Packet epoch int:      " + str(timestampInt) )
        #print( "Tobii Curr/Prev times: " + str(tobiiList[tobiiListPos].timestamp) +  "    " + str(tobiiList[tobiiListPos-1].timestamp) )
        #print( "Diffs (ms):            " + str(diffCurr) +  "    " + str(diffPrev) )

        # Check that we have valid tobii data for this frame
        # If not, do nothing
        if td.rightEyeValid == 0 or td.leftEyeValid == 0:
            return

        ###########################################################################################################
        # Put Tobii and WebGazer gaze estimates into the same coordinate system (normalized screen pixels [0,1], top left is 0,0)
        # Note the chromeDownBar - look for it in the P_1.mp4 videos...
        wgPredX = float(wgPredX + (wgWindowOuterWidth - wgWindowInnerWidth + wgWindowX)) / screenWidthPixels
        wgPredY = float(wgPredY + (wgWindowOuterHeight - wgWindowInnerHeight - chromeDownloadBarHeight + wgWindowY)) / screenHeightPixels

        ###########################################################################################################
        # Compute the difference between the WebGazer prediction and the Tobii prediction as Euclidean screen space distance
        # Let's average the Tobii left/right eye predictions
        tobiiX = float(td.leftScreenGazeX + td.rightScreenGazeX) / 2.0
        tobiiY = float(td.leftScreenGazeY + td.rightScreenGazeY) / 2.0
        diffX = tobiiX - wgPredX
        diffY = tobiiY - wgPredY
        dist = math.sqrt( diffX*diffX + diffY*diffY )
        # Increment counters
        tobiiWGDiffPerVideo = tobiiWGDiffPerVideo + dist
        tobiiWGNumFrames = tobiiWGNumFrames + 1

        ###########################################################################################################
        # Write the difference to a file.
        # This is really disk inefficient...
        with open( diffCSV, 'a') as f:
            f.write( fname + "," + str(timestampInt) + "," + str(dist) + "\n")


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]   


def main():
    
    ###########################################################################################################
    # Start webserver
    #
    listen_address = ''
    listen_port = 8000
    try:
        if len(sys.argv) == 2:
            listen_port = int(sys.argv[1])
        elif len(sys.argv) == 3:
            listen_address = sys.argv[1]
            listen_port = int(sys.argv[2])
        assert 0 <= listen_port <= 65535
    except (AssertionError, ValueError) as e:
        raise ValueError('port must be a number between 0 and 65535')

    tornado.log.enable_pretty_logging()
    args = sys.argv
    args.append("--log_file_prefix=myapp.log")
    tornado.options.parse_command_line(args)
    application = tornado.web.Application(              
        [
            (r'/Webgazer', WebgazerHandler),
            (r'/(.*)', tornado.web.StaticFileHandler, {'path': '.', 'default_filename': ''}),
        ],
        gzip=False,
    )
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(listen_port, listen_address)
    logging.info('Listening on %s:%s' % (listen_address or '[::]' if ':' not in listen_address else '[%s]' % listen_address, listen_port))
    
    # [James]
    # Comment these lines to allow full output to console, e.g., for GET requests
    #logging.getLogger('tornado.access').disabled = True
    #logging.getLogger('tornado.application').disabled = True
    #logging.getLogger('tornado.general').disabled = True
    
    tornado.ioloop.IOLoop.instance().start()

    ###########################################################################################################
    # TODO Spawn Chrome and ask it to go to the webgazerTester.html page ?


if __name__ == '__main__':
    main()