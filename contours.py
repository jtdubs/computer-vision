#!/usr/bin/python

from ctypes         import *
from opencv         import *
from opencv.highgui import *

dilation        = 1
block_size      = 9
param1          = 8
should_dilate   = False
should_equalize = False

def contours(img, storage):
    cvClearMemStorage(storage)
    scanner = cvStartFindContours(img, storage, mode=CV_RETR_TREE, method=CV_CHAIN_APPROX_SIMPLE)
    contour = cvFindNextContour(scanner)
    while contour:
        yield pointee(cast(pointer(contour), CvContour_p))
        contour = cvFindNextContour(scanner)
    del scanner

def quadrangles(img, storage):
    for contour in contours(img, storage):
        if contour.flags & CV_SEQ_FLAG_HOLE and (contour.rect.width*contour.rect.height) >= 100:
            tmp = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, 7)
            tmp = cvApproxPoly(tmp,     sizeof(CvContour), None, CV_POLY_APPROX_DP, 7)
            if tmp.total == 4 and cvCheckContourConvexity(tmp):
                yield contour

def main():
    global dilation, block_size, param1, should_dilate, should_equalize

    capture         = cvCaptureFromCAM(0)
    frame           = cvQueryFrame(capture)
    size            = cvSize(frame.width, frame.height)
    gray            = cvCreateImage(size, 8,  1)
    threshold       = cvCreateImage(size, 8, 1)
    contour_storage = cvCreateMemStorage(0)

    cvNamedWindow('threshold', 1)
    cvNamedWindow('contours', 1)

    while True:
        frame = cvQueryFrame(capture)

        cvCvtColor(frame, gray, CV_BGR2GRAY)
        if should_equalize:
            cvEqualizeHist(gray, gray)

        cvAdaptiveThreshold(gray, threshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, block_size, param1)
        if should_dilate:
            cvDilate(threshold, threshold, iterations=dilation)
        cvShowImage('threshold', threshold)

        for quadrangle in quadrangles(threshold, contour_storage):
            cvDrawContours(frame, quadrangle, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 1, 8)
        cvShowImage('contours', frame)

        k = cvWaitKey(10)
        if   k == 27:       break
        elif k == ord('j'): block_size = max(block_size - 2, 3)
        elif k == ord('k'): block_size = block_size + 2
        elif k == ord('l'): param1 = param1 + 1
        elif k == ord('h'): param1 = max(param1 - 1, 0)
        elif k == ord('a'): dilation = dilation + 1
        elif k == ord('z'): dilation = max(dilation - 1, 1)
        elif k == ord('d'): should_dilate = not should_dilate
        elif k == ord('e'): should_equalize = not should_equalize

        if k <> -1:
            print dilation, block_size, param1, should_dilate, should_equalize

if __name__ == '__main__':
    main()
