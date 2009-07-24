#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
from random      import *
from ctypes      import *
import sys

class DecalIdentifier:
    def __init__(self, debug=True):
        self.debug           = debug
        self.image           = None                                  # input rgb image
        self.gray            = None                                  # grayscale image
        self.edges           = None                                  # output of canny edge detection
        self.rectified       = cvCreateImage(cvSize(100, 100), 8, 3) # holds rectified decal during identification
        self.storage         = cvCreateMemStorage(0)                 # holds the contour/poly/decal coords
        self.intrinsic       = cvCreateMat(3, 3, CV_32FC1)           # camera intrinsic parameters (see cvCalibrateCamera2)
        self.distortion      = cvCreateMat(1, 4, CV_32FC1)           # camera distortion coefficients (see cvCalibrateCamera2)
        self.decal_mat       = cvCreateMat(4, 3, CV_32FC1)           # 3D coordinates of ideal decal corners (unit square in xy-plane)
        self.image_mat       = cvCreateMat(4, 2, CV_32FC1)           # 2D image-space coordinates of detected decal
        self.rotation        = cvCreateMat(1, 3, CV_32FC1)           # calculated rotation matrix for decal (stored as vector)
        self.rotation_matrix = cvCreateMat(3, 3, CV_32FC1)           # calculated rotation matrix for decal (stored as matrix)
        self.translation     = cvCreateMat(1, 3, CV_32FC1)           # calculated translation vector for decal
        self.perspective     = cvCreateMat(3, 3, CV_32FC1)           # calculated perspective transform matrix (used during identification)

        camera_intrinsic  = [600, 0, 320, 0, 600, 240, 0, 0, 1] # from output of tools/calibrate.py
        camera_distortion = [0, 0, 0, 0]                        # from output of tools/calibrate.py

        for x in range(0, 3):
            for y in range(0, 3):
                self.intrinsic[x, y] = camera_intrinsic[(x*3)+y]

        for x in range(0, 4):
            self.distortion[0, x] = camera_distortion[x]

        for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            self.decal_mat[i,0], self.decal_mat[i,1], self.decal_mat[i,2] = x, y, 0

    def get_decals(self, image):
        self.image = image

        if self.gray == None:
            size       = cvSize(self.image.width, self.image.height)
            self.gray  = cvCreateImage(size, 8, 1)
            self.edges = cvCreateImage(size, 8, 1)

        cvCvtColor(self.image, self.gray, CV_BGR2GRAY) # convert to gray-scale
        cvCanny(self.gray, self.edges, 805, 415, 5)    # edge detect (hand tuned w/ canny.py)
        cvDilate(self.edges, self.edges, iterations=1) # dilate edges to help bridge small gaps in contours

        contours = self._find_contours(self.edges, self.storage) # find contours
        polys    = list(self._find_polys(contours))              # find polys in contours
        decals   = self._find_decals(polys)                      # find decals in polys

        for i, decal in enumerate(decals):
            # calculate more precise, sub-pixel corner positions
            corners = [CvPoint2D32f(p.x, p.y) for p in decal.asarray(CvPoint)]
            corners = cvFindCornerSubPix(self.gray, corners, CvSize(5, 5), CvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.001))

            # identify decal value and orientation
            decal_value, decal_orientation = self._identify_decal(i, corners, decal)

            # apply orientation to corners so we get the correct rotation matrix
            corners = corners[4-decal_orientation:] + corners[:4-decal_orientation]

            for j, corner in enumerate(corners):
                self.image_mat[j,0], self.image_mat[j,1] = corner.x, corner.y

            # calculate rotation and translation of decal
            cvFindExtrinsicCameraParams2(self.decal_mat, self.image_mat, self.intrinsic, self.distortion, self.rotation, self.translation)
            cvRodrigues2(self.rotation, self.rotation_matrix)

            # combine rotation and translation into an opengl modelview matrix
            modelview = [0.0] * 16
            for x in range(0, 3):
                for y in range(0, 3):
                    modelview[(y*4)+x] = self.rotation_matrix[x,y]
            modelview[12] = self.translation[0,0];
            modelview[13] = self.translation[0,1];
            modelview[14] = self.translation[0,2];
            modelview[15] = 1.0;

            yield (modelview, decal_value)

    def _identify_decal(self, offset, corners, decal):
        # warp decal to a flat, 100x100 square
        b = cvBoundingRect(decal, 0)
        if self.debug:
            cvRectangle(self.image, cvPoint(b.x,b.y), cvPoint(b.x+b.width, b.y+b.height), CV_RGB(255,0,255), 2, 8, 0)
        cvGetPerspectiveTransform(corners, as_c_array([CvPoint2D32f(0,0), CvPoint2D32f(0,100), CvPoint2D32f(100,100), CvPoint2D32f(100,0)], None, CvPoint2D32f), self.perspective)
        cvWarpPerspective(self.image, self.rectified, self.perspective, CV_WARP_FILL_OUTLIERS)

        # average each of the 4 quadrants
        c = [None] * 4
        cvSetImageROI(self.rectified, cvRect(25, 25, 25, 25)); c[0] = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(50, 25, 25, 25)); c[1] = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(50, 50, 25, 25)); c[2] = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(25, 50, 25, 25)); c[3] = cvAvg(self.rectified)

        # re-orient to put darkest quadrant in top-left (first position)
        c_avg = [None] * 4
        for i in range(0, 4):
            c_avg[i] = (c[i].val[0] + c[i].val[1] + c[i].val[2]) / 3
        orientation = c_avg.index(min(c_avg))
        c = c[orientation:] + c[:orientation]

        if self.debug:
            # render quadrants to self.image for debugging purposes
            cvSetImageROI(self.image, cvRect(offset*50,    0,    25, 25)); cvSet(self.image, c[0])
            cvSetImageROI(self.image, cvRect(offset*50+25, 0,    25, 25)); cvSet(self.image, c[1])
            cvSetImageROI(self.image, cvRect(offset*50+25, 0+25, 25, 25)); cvSet(self.image, c[2])
            cvSetImageROI(self.image, cvRect(offset*50,    0+25, 25, 25)); cvSet(self.image, c[3])

        # stretch color range between perceived black and perceived white
        cvSetImageROI(self.rectified, cvRect(0,   0, 100, 25)); white1 = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(0,  75, 100, 25)); white2 = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(0,  25,  25, 75)); white3 = cvAvg(self.rectified)
        cvSetImageROI(self.rectified, cvRect(75, 25, 100, 75)); white4 = cvAvg(self.rectified)
        white = [(white1.val[i]*4 + white2.val[i]*4 + white3.val[i]*2 + white4.val[i]*2) / 12 for i in range(0, 3)]
        black = [c[0].val[i] for i in range(0, 3)]
        diff  = [white[j] - black[j] for j in range(0, 3)]

        for i in range(0, 4):
            for j in range(0, 3):
                c[i].val[j] = max(0, min(255, (c[i].val[j] - black[j]) * 255 / diff[j]))

        if self.debug:
            # render quadrants to self.image for debugging purposes
            cvSetImageROI(self.image, cvRect(offset*50,    50,    25, 25)); cvSet(self.image, c[0])
            cvSetImageROI(self.image, cvRect(offset*50+25, 50,    25, 25)); cvSet(self.image, c[1])
            cvSetImageROI(self.image, cvRect(offset*50+25, 50+25, 25, 25)); cvSet(self.image, c[2])
            cvSetImageROI(self.image, cvRect(offset*50,    50+25, 25, 25)); cvSet(self.image, c[3])
            cvSetImageROI(self.image, cvRect(offset*50+50, 50,    25, 25)); cvSet(self.image, CV_RGB(black[0], black[1], black[2]))
            cvSetImageROI(self.image, cvRect(offset*50+50, 50+25, 25, 25)); cvSet(self.image, CV_RGB(white[0], white[1], white[2]))

        # determine color of quadrants, and therefore decal value
        decal_value = 0
        for i in range(1, 4):
            channels    = [c[i].val[j] for j in range(0, 3)]
            sorted      = list(channels)
            sorted.sort()
            max_channel = channels.index(sorted[2])
            # hand-tuned color thresholds based on experimentation in my living rooms lighting conditions... :-)
            if max_channel == 0:   # blue
                value = (max_channel+1) if (sorted[2] / float(max(1, sorted[1]))) >= 1.2 and (sorted[2] - sorted[1]) >= 20 else 0
            elif max_channel == 1: # green
                value = (max_channel+1) if (sorted[2] / float(max(1, sorted[1]))) >= 1.5 and (sorted[2] - sorted[1]) >= 30 else 0
            elif max_channel == 2: # red:
                value = (max_channel+1) if (sorted[2] / float(max(1, sorted[1]))) >= 2.0 and (sorted[2] - sorted[1]) >= 40 else 0
            decal_value = (decal_value * 4) + value

            c[i].val[(max_channel+0)%3] = 0 if value == 0 else 255
            c[i].val[(max_channel+1)%3] = 0
            c[i].val[(max_channel+2)%3] = 0

        if self.debug:
            # render quadrants to self.image for debugging purposes
            cvSetImageROI(self.image, cvRect(offset*50,    100,    25, 25)); cvSet(self.image, CV_RGB(0,0,0))
            cvSetImageROI(self.image, cvRect(offset*50+25, 100,    25, 25)); cvSet(self.image, c[1])
            cvSetImageROI(self.image, cvRect(offset*50+25, 100+25, 25, 25)); cvSet(self.image, c[2])
            cvSetImageROI(self.image, cvRect(offset*50,    100+25, 25, 25)); cvSet(self.image, c[3])

        cvResetImageROI(self.image)
        cvResetImageROI(self.rectified)

        return (decal_value, orientation)

    def _find_contours(self, img, storage):
        cvClearMemStorage(storage)
        scanner = cvStartFindContours(img, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE)
        contour = cvFindNextContour(scanner)
        while contour:
            yield pointee(cast(pointer(contour), CvContour_p))
            contour = cvFindNextContour(scanner)
        del scanner

    def _find_polys(self, contours):
        for contour in contours:
            hole = contour.flags & CV_SEQ_FLAG_HOLE
            if hole and (contour.rect.width*contour.rect.height) > 200:
                poly = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, max(contour.rect.width,contour.rect.height)/8)
                if self.debug:
                    cvDrawContours(self.image, poly, CV_RGB(255,0,0), CV_RGB(255,0,0), 0, 1, 8)
                yield poly

    def _find_decals(self, polys):
        pointers = dict([(addressof(p), p) for p in polys])
        decal_to_inner = { }
        inner_to_decal = { }

        for decal in polys:
            # decals must be quads
            if decal.total <> 4:
                continue

            # find all the polys contained within this decal
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

            # decals must have atleast one inner poly, but no more than five
            if 1 <= len(inner_polys) <= 5:
                valid_decal = True

                if inner_to_decal.has_key(addressof(decal)):
                    # we are inside another decal, so the outer decal was a false positive.  delete it.
                    del decal_to_inner[inner_to_decal[addressof(decal)]]
                else:
                    for inner in inner_polys:
                        if decal_to_inner.has_key(addressof(inner)):
                            # one of our inner polys is a decal, so we are a false positive.  skip ourselves.
                            valid_decal = False

                if valid_decal:
                    # valid decal found, remember it
                    decal_to_inner[addressof(decal)] = inner_polys
                    for inner in inner_polys:
                        inner_to_decal[addressof(inner)] = addressof(decal)

        # return all the decals
        return [pointers[addr] for addr in decal_to_inner.keys()]
