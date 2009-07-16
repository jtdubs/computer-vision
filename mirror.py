#!/usr/bin/python

from opencv         import *
from opencv.highgui import *

def main():
    capture = cvCaptureFromCAM(0)
    frame   = cvQueryFrame(capture)
    size    = cvSize(frame.width, frame.height)
    copy    = cvCreateImage(size, 8, 3)

    cvNamedWindow('frame', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)
        cvShowImage('frame', copy)

        k = cvWaitKey(10)
        if k == 27: break

if __name__ == '__main__':
    main()
