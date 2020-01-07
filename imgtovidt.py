import cv2
import numpy as np
import os
from os.path import isfile, join
import time
from datetime import datetime
import fnmatch
import shutil
import threading
import subprocess
import signal

t = time.localtime()
#b1 = ['/home/pi/src/csrc/mediatronix_ipcam/CameraImageRx', '-c', '/home/pi/src/csrc/mediatronix_ipcam/config.json']
#b2 = 'rm -r /dev/shm/Images/*H00*.jpg'
#c1 = "rm -r /dev/shm/Images/*"
#try:
#    subprocess.run(b1, timeout=150)
#except subprocess.TimeoutExpired:
#    print('KILLED')
def generate_video():
    pathIn = './IMAGES/INCIDENTS1/INCI2/'
    fps = 6
    print("START1")
    #os.system(b2)
    print("START2")
    timestamp = time.strftime('%b-%d-%y_%H%M', t)
    pathOut = './IMAGES%s.avi'%timestamp
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: x[5:-4])
    for i in range(len(files)):
        filename = pathIn +files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'H264'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
generate_video()
#os.system(c1)
#out.release()



