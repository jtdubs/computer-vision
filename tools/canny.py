#!/usr/bin/python

from ctypes         import *
from opencv         import *
from opencv.highgui import *

def main():
    capture         = cvCaptureFromCAM(0)
    frame           = cvQueryFrame(capture)
    size            = cvSize(frame.width, frame.height)
    copy            = cvCreateImage(size, 8, 3)
    gray            = cvCreateImage(size, 8, 1)
    edges           = cvCreateImage(size, 8, 1)

    t1, t2, ap = 805, 415, 5

    cvNamedWindow('frame', 1)
    cvNamedWindow('edges', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)

        cvCvtColor(copy, gray, CV_BGR2GRAY)
        cvCanny(gray, edges, t1, t2, ap)

        cvShowImage('edges', edges)
        cvShowImage('frame', copy)

        k = cvWaitKey(10)
        if   k == 27:       break
        elif k == ord('a'): t1 = min(t1 + 5, 2048)
        elif k == ord('z'): t1 = max(t1 - 5, 1)
        elif k == ord('s'): t2 = min(t2 + 5, 2048)
        elif k == ord('x'): t2 = max(t2 - 5, 1)
        elif k == ord('d'): ap = min(ap + 2, 7)
        elif k == ord('c'): ap = max(ap - 2, 1)

        if k <> -1: print t1, t2, ap

if __name__ == '__main__':
    main()
