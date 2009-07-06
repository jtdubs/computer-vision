#!/usr/bin/python

from math           import *
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
            quad = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, 7)
            quad = cvApproxPoly(quad,    sizeof(CvContour), None, CV_POLY_APPROX_DP, 7)
            if quad.total == 4 and cvCheckContourConvexity(quad):
                yield quad

def quad_to_points(quad):
    return [(p.x, p.y) for p in quad.asarray(CvPoint)]

def quad_angle(quad):
    [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] = quad
    a1, a2, a3, a4 = atan2(y2-y1,x2-x1)+pi%pi,      atan2(y3-y2,x3-x2)+pi%pi,      atan2(y4-y3,x4-x3)+pi%pi,      atan2(y1-y4,x1-x4)+pi%pi
    a1, a2, a3, a4 = (a1 if a1<=(pi/2) else a1-pi), (a2 if a2<=(pi/2) else a2-pi), (a3 if a3<=(pi/2) else a3-pi), (a4 if a4<=(pi/2) else a4-pi)
    return min([a1, a2, a3, a4], key=abs) * 180 / pi

def chessboards(quads):
    quads = [quad_to_points(q) for q in quads]
    return [(quad_angle(q), q) for q in quads]

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

        quads = list(quadrangles(threshold, contour_storage))
        for quadrangle in quads:
            cvDrawContours(frame, quadrangle, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 1, 8)
        cvShowImage('contours', frame)

        boards = chessboards(quads)
        if boards:
            print boards

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
