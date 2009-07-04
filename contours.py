#!/usr/bin/python

from ctypes         import *
from math           import *
from opencv         import *
from opencv.highgui import *

scale = 1.3

def main():
    capture   = cvCaptureFromCAM(0)
    frame     = cvQueryFrame(capture)
    size      = cvSize(frame.width, frame.height)
    gray      = cvCreateImage(size, 8,  1)
    threshold = cvCreateImage(size, 8, 1)
    storage   = cvCreateMemStorage(0)
    storage2  = cvCreateMemStorage(0)

    cvNamedWindow('gray', 1)
    cvNamedWindow('threshold', 1)
    cvNamedWindow('contours', 1)

    dilation = 1
    block_size = 9
    param1 = 8
    should_dilate = False
    should_equalize = False

    while True:
        frame = cvQueryFrame(capture)

        cvCvtColor(frame, gray, CV_BGR2GRAY)
        if should_equalize:
            cvEqualizeHist(gray, gray)
        cvShowImage('gray', gray)

        cvAdaptiveThreshold(gray, threshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, block_size, param1)
        cvRectangle(threshold, CvPoint(0, 0), CvPoint(639, 479), CV_RGB(0,0,0), 3, 8)
        if should_dilate:
            cvDilate(threshold, threshold, iterations=dilation)
        cvShowImage('threshold', threshold)

        cvClearMemStorage(storage)
        scanner = cvStartFindContours(threshold, storage, mode=CV_RETR_TREE, method=CV_CHAIN_APPROX_SIMPLE)
        seq     = cvFindNextContour(scanner)
        while seq:
            contour = pointee(cast(pointer(seq), CvContour_p))
            if contour.flags & CV_SEQ_FLAG_HOLE and (contour.rect.width*contour.rect.height) >= 100:
                cvClearMemStorage(storage2)
                tmp = cvApproxPoly(contour, sizeof(CvContour), storage2, CV_POLY_APPROX_DP, 7)
                tmp = cvApproxPoly(tmp,     sizeof(CvContour), storage2, CV_POLY_APPROX_DP, 7)
                if tmp.total == 4 and cvCheckContourConvexity(tmp):
                    cvDrawContours(frame, seq, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 1, 8)

            seq = cvFindNextContour(scanner)
        del scanner
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

        # 3, 9, 5, True,  False
        # 3, 9, 7, False, False
        if k <> -1:
            print dilation, block_size, param1, should_dilate, should_equalize

if __name__ == '__main__':
    main()
