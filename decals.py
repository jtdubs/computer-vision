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

def polys(contours):
    for contour in contours:
        hole = contour.flags & CV_SEQ_FLAG_HOLE
        if hole:
            poly = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, contour.rect.width/10)
            if cvCheckContourConvexity(poly):
                yield poly

def decals(polys):
    for decal in polys:
        if decal.total == 4:
            inner_polys = []
            for inner_poly in polys:
                if pointer(inner_poly) <> pointer(decal):
                    inside = True
                    for pt in inner_poly.asarray(CvPoint):
                        pt = CvPoint2D32f(pt.x, pt.y)
                        if cvPointPolygonTest(decal, pt, 0) <= 0:
                            inside = False
                    if inside:
                        inner_polys.append(inner_poly)
            if len(inner_polys) == 1:
                yield (decal, inner_polys[0].total)

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

        ps = list(polys(contours(edges, contour_storage)))
        # for poly in ps:
        #     cvDrawContours(copy, poly, CV_RGB(255,0,0), CV_RGB(255,0,0), 0, 2, 8)
        for (decal, n) in decals(ps):
            if n == 3:
                cvDrawContours(copy, decal, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 2, 8)
            elif n == 4:
                cvDrawContours(copy, decal, CV_RGB(0,0,255), CV_RGB(0,0,255), 0, 2, 8)

        cvShowImage('contours', copy)

        k = cvWaitKey(10)
        if  k == 27: break

if __name__ == '__main__':
    main()
