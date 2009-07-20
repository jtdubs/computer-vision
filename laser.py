#!/usr/bin/python

from opencv         import *
from opencv.highgui import *

def main():
    capture = cvCaptureFromCAM(0)
    frame   = cvQueryFrame(capture)
    size    = cvSize(frame.width, frame.height)
    copy    = cvCreateImage(size, 8, 3)
    r       = cvCreateImage(size, 8, 1)
    g       = cvCreateImage(size, 8, 1)
    b       = cvCreateImage(size, 8, 1)
    mask    = cvCreateImage(size, 8, 1)
    mask2   = cvCreateImage(size, 8, 1)
    laser   = cvCreateImage(size, 8, 1)
    max     = cvCreateImage(size, 8, 1)
    min     = cvCreateImage(size, 8, 1)
    diff    = cvCreateImage(size, 8, 1)

    cvNamedWindow('laser', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)
        cvSplit(copy, b, g, r)

        # assert |r-g| > 50
        cvAbsDiff(r, g, diff)
        cvCmpS(diff, 50, mask, CV_CMP_GT)
        # cvAnd(mask, mask2, mask)

        # assert |r-b| > 50
        cvAbsDiff(r, b, diff)
        cvCmpS(diff, 50, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert r > g
        cvCmp(r, g, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert r > b
        cvCmp(r, b, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert r > 200
        cvCmpS(r, 200, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        cvAddWeighted(mask, 1.0, laser, 0.95, 0.0, laser)
        cvShowImage('laser', laser)

        k = cvWaitKey(10)
        if k == 27: break

if __name__ == '__main__':
    main()
