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

# at angles, i sometimes mistake squares for nearly-square octagons (check convexity defects to weed out)
def quadrangles(img, storage, frame=None):
    for contour in contours(img, storage):
        hole  = contour.flags & CV_SEQ_FLAG_HOLE
        area  = contour.rect.width * contour.rect.height
        if hole and area >= 225:
            quad = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, 6)
            quad = cvApproxPoly(quad,    sizeof(CvContour), None, CV_POLY_APPROX_DP, 6)
            if cvCheckContourConvexity(quad) and quad.total == 4:
                yield quad
#            elif area > 2000 and quad.total == 8:
#                box = cvMinAreaRect2(quad)
#                box_area  = box.size.width * box.size.height
#                quad_area = abs(cvContourArea(quad)) * 2
#                if quad_area > 1000:
#                    print box_area, quad_area
#                    cvDrawContours(frame, quad, CV_RGB(255,0,0), CV_RGB(255,0,0), 0, 1, 8)
#                    for p in quad.asarray(CvPoint):
#                        cvCircle(frame, p, 3, CV_RGB(0, 0, 255), 1)
#                    if abs(box_area - quad_area) <= (box_area / 10.0):
#                        print quad

def quad_to_points(quad):
    return tuple([(p.x, p.y) for p in quad.asarray(CvPoint)])

def quad_angle(quad):
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = quad
    a1, a2, a3, a4 = atan2(y2-y1,x2-x1)+pi%pi,      atan2(y3-y2,x3-x2)+pi%pi,      atan2(y4-y3,x4-x3)+pi%pi,      atan2(y1-y4,x1-x4)+pi%pi
    a1, a2, a3, a4 = (a1 if a1<=(pi/2) else a1-pi), (a2 if a2<=(pi/2) else a2-pi), (a3 if a3<=(pi/2) else a3-pi), (a4 if a4<=(pi/2) else a4-pi)
    return min([a1, a2, a3, a4], key=abs) * 180 / pi

def quad_center(quad):
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = quad
    return ((x1+x2+x3+x4)/4.0, (y1+y2+y3+y4)/4.0)

def quad_size(quad):
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = quad
    return (max(x1, x2, x3, x4) - min(x1, x2, x3, x4), max(y1, y2, y3, y4) - min(y1, y2, y3, y4))

def quad_groups(quads):
    quads = [quad_to_points(q) for q in quads]
    quads = [(quad_angle(q), quad_center(q), quad_size(q), q) for q in quads]
    return [g for g in group_quads(quads, []) if len(g) >= 3]

def quad_matches(q1, q2):
    (a1, (cx1, cy1), (sx1, sy1), q1) = q1
    (a2, (cx2, cy2), (sx2, sy2), q2) = q2

    return abs(a1-a2) < 15 \
       and abs(cx1-cx2) < max(sx1,sx2)*2.0 \
       and abs(cy1-cy2) < max(sy1,sy2)*2.0 \
       and 1/2.0 < sx1/float(sx2) < 2.0 \
       and 1/2.0 < sy1/float(sy2) < 2.0

def group_quads(quads, components):
    for quad in quads:
        matches = []

        for c in components:
            for q in c:
                if quad_matches(quad, q):
                    matches.append(c)

        for c in matches:
            try:
                components.remove(c)
            except:
                pass

        components.append(set([quad]).union(*matches))

    return components

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

        quads = list(quadrangles(threshold, contour_storage, frame=frame))
        for quadrangle in quads:
            cvDrawContours(frame, quadrangle, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 1, 8)

        # intensities = [255, 128, 64]
        # colors      = [CV_RGB(x,y,z) for x in intensities for y in intensities for z in intensities]
        # for color, quads in zip(colors, quad_groups(quads)):
        #     for (_, _, _, points) in quads:
        #         cvPolyLine(frame, [[CvPoint(x,y) for (x,y) in list(points)]], True, color, 2, 8)

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
