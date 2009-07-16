#!/usr/bin/python

from math           import *
from ctypes         import *
from opencv         import *
from opencv.highgui import *

def contours(img, storage):
    cvClearMemStorage(storage)
    scanner = cvStartFindContours(img, storage, mode=CV_RETR_TREE, method=CV_CHAIN_APPROX_SIMPLE)
    contour = cvFindNextContour(scanner)
    while contour:
        yield pointee(cast(pointer(contour), CvContour_p))
        contour = cvFindNextContour(scanner)
    del scanner

def find_decals(contour, storage):
    if contour is not None:
        hole = contour.flags & CV_SEQ_FLAG_HOLE
        if not hole:
            cvClearMemStorage(storage)
            decal = cvApproxPoly(contour, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 7)
            decal = cvApproxPoly(decal,   sizeof(CvContour), None,   CV_POLY_APPROX_DP, 7)
            if decal.total == 4 and cvCheckContourConvexity(decal):
                marker = pointee(cast(contour.v_next, CvContour_p))
                if marker is not None:
                    cvClearMemStorage(storage)
                    marker_poly = cvApproxPoly(marker,      sizeof(CvContour), storage, CV_POLY_APPROX_DP, 7)
                    marker_poly = cvApproxPoly(marker_poly, sizeof(CvContour), None,    CV_POLY_APPROX_DP, 7)
                    if marker_poly.total == 4 and cvCheckContourConvexity(marker_poly):
                        hole = pointee(cast(marker.v_next, CvContour_p))
                        cvClearMemStorage(storage)
                        hole_poly = cvApproxPoly(hole,      sizeof(CvContour), storage, CV_POLY_APPROX_DP, 7)
                        hole_poly = cvApproxPoly(hole_poly, sizeof(CvContour), None,    CV_POLY_APPROX_DP, 7)
                        if hole_poly.total == 4 and cvCheckContourConvexity(hole_poly):
                            yield contour

        h_next = pointee(cast(contour.h_next, CvContour_p))
        v_next = pointee(cast(contour.v_next, CvContour_p))

        for d in find_decals(v_next, storage): yield d
        for d in find_decals(h_next, storage): yield d

def main():
    capture         = cvCaptureFromCAM(0)
    frame           = cvQueryFrame(capture)
    size            = cvSize(frame.width, frame.height)
    copy            = cvCreateImage(size, 8, 3)
    gray            = cvCreateImage(size, 8, 1)
    threshold       = cvCreateImage(size, 8, 1)
    contour_storage = cvCreateMemStorage(0)
    decal_storage   = cvCreateMemStorage(0)

    cvNamedWindow('threshold', 1)
    cvNamedWindow('contours', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)

        cvCvtColor(copy, gray, CV_BGR2GRAY)

        cvAdaptiveThreshold(gray, threshold, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, 8)
        cvShowImage('threshold', threshold)

        contour = list(contours(threshold, contour_storage))[0]
        for decal in find_decals(contour, decal_storage):
            cvDrawContours(copy, decal, CV_RGB(0,255,0), CV_RGB(0,0,255), 5, 1, 8)

        cvShowImage('contours', copy)

        k = cvWaitKey(10)
        if k == 27: break

if __name__ == '__main__':
    main()
