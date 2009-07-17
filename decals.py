#!/usr/bin/python

from ctypes         import *
from opencv         import *
from opencv.highgui import *

def contours(img, storage):
    cvClearMemStorage(storage)
    scanner = cvStartFindContours(img, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE)
    contour = cvFindNextContour(scanner)
    while contour:
        yield pointee(cast(pointer(contour), CvContour_p))
        contour = cvFindNextContour(scanner)
    del scanner

def quads(contours, storage):
    for contour in contours:
        hole = contour.flags & CV_SEQ_FLAG_HOLE
        if not hole:
            cvClearMemStorage(storage)
            quad = cvApproxPoly(contour, sizeof(CvContour), storage, CV_POLY_APPROX_DP, contour.rect.width/10)
            if quad.total == 4 and cvCheckContourConvexity(quad):
                yield contour

def decals(quads):
    for decal in quads:
        inside_count = 0
        for marker in quads:
            if pointer(marker) <> pointer(decal):
                inside = True
                for pt in marker.asarray(CvPoint):
                    pt = CvPoint2D32f(pt.x, pt.y)
                    if cvPointPolygonTest(decal, pt, 0) <= 0:
                        inside = False
                if inside:
                    inside_count = inside_count + 1
        if inside_count == 1:
            yield decal

def main():
    capture         = cvCaptureFromCAM(0)
    frame           = cvQueryFrame(capture)
    size            = cvSize(frame.width, frame.height)
    copy            = cvCreateImage(size, 8, 3)
    gray            = cvCreateImage(size, 8, 1)
    edges           = cvCreateImage(size, 8, 1)
    contour_storage = cvCreateMemStorage(0)
    quad_storage    = cvCreateMemStorage(0)

    cvNamedWindow('edges', 1)
    cvNamedWindow('contours', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)
        cvCvtColor(copy, gray, CV_BGR2GRAY)
        cvCanny(gray, edges, 805, 415, 5) # hand tuned w/ canny.py
        cvDilate(edges, edges, iterations=1)
        cvShowImage('edges', edges)

        qs = list(quads(contours(edges, contour_storage), quad_storage))
        # for quad in qs:
        #     cvDrawContours(copy, quad, CV_RGB(255,0,0), CV_RGB(255,0,0), 0, 2, 8)
        for decal in decals(qs):
            cvDrawContours(copy, decal, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 2, 8)

        cvShowImage('contours', copy)

        k = cvWaitKey(10)
        if   k == 27:       break

if __name__ == '__main__':
    main()

#             marker = pointee(cast(contour.v_next, CvContour_p))
#             if marker is not None:
#                 cvClearMemStorage(storage)
#                 marker_poly = cvApproxPoly(marker,      sizeof(CvContour), storage, CV_POLY_APPROX_DP, 7)
#                 marker_poly = cvApproxPoly(marker_poly, sizeof(CvContour), None,    CV_POLY_APPROX_DP, 7)
#                 if marker_poly.total == 4 and cvCheckContourConvexity(marker_poly):
#                     yield contour

#                        hole = pointee(cast(marker.v_next, CvContour_p))
#                        cvClearMemStorage(storage)
#                        hole_poly = cvApproxPoly(hole,      sizeof(CvContour), storage, CV_POLY_APPROX_DP, 7)
#                        hole_poly = cvApproxPoly(hole_poly, sizeof(CvContour), None,    CV_POLY_APPROX_DP, 7)
#                        if hole_poly.total == 4 and cvCheckContourConvexity(hole_poly):
#                            yield contour

