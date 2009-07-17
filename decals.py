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
    capture            = cvCaptureFromCAM(0)
    frame              = cvQueryFrame(capture)
    size               = cvSize(frame.width, frame.height)
    copy               = cvCreateImage(size, 8, 3)
    gray               = cvCreateImage(size, 8, 1)
    edges              = cvCreateImage(size, 8, 1)
    storage            = cvCreateMemStorage(0)
    decal_mat          = cvCreateMat(4, 3, CV_32FC1)
    image_mat          = cvCreateMat(4, 2, CV_32FC1)
    intrinsic          = cvCreateMat(3, 3, CV_32FC1)
    distortion         = cvCreateMat(1, 4, CV_32FC1)
    rotation           = cvCreateMat(1, 3, CV_32FC1)
    rotation_matrix    = cvCreateMat(3, 3, CV_32FC1)
    gl_rotation_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    translation        = cvCreateMat(1, 3, CV_32FC1)

    ci = [682.80694580078125, 0.0, 331.3616943359375, 0.0, 631.85980224609375, 210.08140563964844, 0.0, 0.0, 1.0]
    cd = [0.34955242276191711, -0.70636618137359619, -0.013230122625827789, 0.0091487327590584755]
    for x in range(0, 3):
        for y in range(0, 3):
            intrinsic[x, y] = ci[(x*3)+y]
    for x in range(0, 4):
        distortion[0, x] = cd[x]
    for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
        decal_mat[i,0], decal_mat[i,1], decal_mat[i,2] = x, y, 0

    cvNamedWindow('edges', 1)
    cvNamedWindow('contours', 1)

    while True:
        frame = cvQueryFrame(capture)
        cvFlip(frame, copy, 1)
        cvCvtColor(copy, gray, CV_BGR2GRAY)
        cvCanny(gray, edges, 805, 415, 5) # hand tuned w/ canny.py
        cvDilate(edges, edges, iterations=1)
        cvShowImage('edges', edges)

        ps = list(polys(contours(edges, storage)))
        # for poly in ps:
        #     cvDrawContours(copy, poly, CV_RGB(255,0,0), CV_RGB(255,0,0), 0, 2, 8)
        for (decal, n) in decals(ps):
            color = CV_RGB(0,255,0) if n == 3 else CV_RGB(0,0,255)
            cvDrawContours(copy, decal, color, color, 0, 2, 8)

            ps = [CvPoint2D32f(p.x, p.y) for p in decal.asarray(CvPoint)]
            cvFindCornerSubPix(gray, ps, CvSize(2, 2), CvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01))
            for i, p in enumerate(ps):
                image_mat[i,0], image_mat[i,1] = p.x, p.y

            cvFindExtrinsicCameraParams2(decal_mat, image_mat, intrinsic, distortion, rotation, translation)
            cvRodrigues2(rotation, rotation_matrix)
            for x in range(0,3):
                for y in range(0,3):
                    gl_rotation_matrix[(y*4)+x] = rotation_matrix[x,y]

        cvShowImage('contours', copy)

        k = cvWaitKey(10)
        if  k == 27: break

if __name__ == '__main__':
    main()
