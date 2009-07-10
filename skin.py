#!/usr/bin/python
# -*- coding: utf-8 -*-

from math           import *
from ctypes         import *
from opencv         import *
from opencv.highgui import *

def main():
    capture         = cvCaptureFromCAM(0)
    frame           = cvQueryFrame(capture)
    size            = cvSize(frame.width, frame.height)
    copy            = cvCreateImage(size, 8, 3)
    r               = cvCreateImage(size, 8, 1)
    g               = cvCreateImage(size, 8, 1)
    b               = cvCreateImage(size, 8, 1)
    bgr             = cvCreateImage(size, 8, 3)
    sum             = cvCreateImage(size, 8, 1)
    max             = cvCreateImage(size, 8, 1)
    min             = cvCreateImage(size, 8, 1)
    diff            = cvCreateImage(size, 8, 1)
    mask            = cvCreateImage(size, 8, 1)
    mask2           = cvCreateImage(size, 8, 1)
    dist            = cvCreateImage(size, 32, 1)
    skin            = cvCreateImage(size, 8, 3)

    cvNamedWindow('frame', 1)
    cvNamedWindow('rgb',   1)
    cvNamedWindow('skin',  1)
    cvNamedWindow('mask',  1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)
        cvShowImage('frame', copy)

        # build normalized rgb image
        cvSplit(copy, b, g, r)
        cvAddWeighted(r, 1/3.0, g, 1/3.0, 0.0, sum)
        cvAddWeighted(sum, 1.0, b, 1/3.0, 0.0, sum)
        cvSetImageCOI(copy, 1); cvDiv(r, sum, r, 255/3)
        cvSetImageCOI(copy, 2); cvDiv(g, sum, g, 255/3)
        cvSetImageCOI(copy, 3); cvDiv(b, sum, b, 255/3)
        cvSetImageCOI(copy, 0)
        cvMerge(b, g, r, None, bgr)
        cvShowImage('rgb', bgr)

        # assert rgb in range (20,40,95) - (255,255,255)
        cvInRangeS(bgr, cvScalar(20, 40, 95), cvScalar(255, 255, 255), mask)

        # assert max(r,g,b) - min(r,g,b) > 15
        cvMax(r, g, max); cvMax(max, b, max)
        cvMin(r, g, min); cvMin(max, b, min)
        cvSub(max, min, diff)
        cvCmpS(diff, 15, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert |r-g| > 15
        cvAbsDiff(r, g, diff)
        cvCmpS(diff, 15, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert r > g
        cvCmp(r, g, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # assert r > b
        cvCmp(r, b, mask2, CV_CMP_GT)
        cvAnd(mask, mask2, mask)

        # dilate mask
        cvDilate(mask, mask, None, 3)
        # cvDistTransform(mask, dist, CV_DIST_L2, 5)
        # cvCmpS(dist, 3.0, mask, CV_CMP_GT)
        # cvAdaptiveThreshold(mask2, mask, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 5)
        # cvNot(mask, mask)
        # cvSmooth(mask, mask2, CV_GAUSSIAN, 3)
        # cvThreshold(mask2, mask, 255, 255, CV_THRESH_BINARY)
        cvShowImage('mask', mask)

        # apply mask to frame
        cvSetZero(skin)
        cvCopy(copy, skin, mask)
        # cvNot(mask, mask)
        # cvSet(skin, cvScalar(255, 255, 0), mask)
        cvShowImage('skin', skin)

        k = cvWaitKey(10)
        if k == 27: break

if __name__ == '__main__':
    main()
