#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
from random      import *
from ctypes      import *
import sys

class AugmentedReality:
    def __init__(self):
        self.init_glut()
        self.init_cv()
        self.init_tracker()

    def init_glut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutCreateWindow('Augmented Reality')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_COLOR_MATERIAL)
        glutReshapeFunc(self.on_reshape)
        glutDisplayFunc(self.on_display)
        glutKeyboardFunc(self.on_key)
        glutIdleFunc(self.on_idle)

        glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [5.0, 3.0, -5.0, 1.0])
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHTING)

        self.frame_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.frame_texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    def init_cv(self):
        self.capture    = cvCaptureFromCAM(0)
        self.frame      = cvQueryFrame(self.capture)
        size            = cvSize(self.frame.width, self.frame.height)
        self.copy       = cvCreateImage(size, 8, 3)
        self.temp       = cvCreateImage(cvSize(100, 100), 8, 3)
        self.gray       = cvCreateImage(size, 8, 1)
        self.edges      = cvCreateImage(size, 8, 1)
        self.storage    = cvCreateMemStorage(0)
        self.intrinsic  = cvCreateMat(3, 3, CV_32FC1)
        self.distortion = cvCreateMat(1, 4, CV_32FC1)

        ci = [600, 0, 320, 0, 600, 240, 0, 0, 1]
        cd = [0, 0, 0, 0]

        for x in range(0, 3):
            for y in range(0, 3):
                self.intrinsic[x, y] = ci[(x*3)+y]

        for x in range(0, 4):
            self.distortion[0, x] = cd[x]

    def init_tracker(self):
        self.decal_mat       = cvCreateMat(4, 3, CV_32FC1)
        self.image_mat       = cvCreateMat(4, 2, CV_32FC1)
        self.rotation        = cvCreateMat(1, 3, CV_32FC1)
        self.rotation_matrix = cvCreateMat(3, 3, CV_32FC1)
        self.translation     = cvCreateMat(1, 3, CV_32FC1)
        self.adjust_src      = cvCreateMat(1, 3, CV_32FC1)
        self.adjust_dst      = cvCreateMat(1, 3, CV_32FC1)
        self.perspective     = cvCreateMat(3, 3, CV_32FC1)
        self.decals          = []

        for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            self.decal_mat[i,0], self.decal_mat[i,1], self.decal_mat[i,2] = x, y, 0

    def on_reshape(self, w, h):
        w, h = 640, 480

        glViewport(0, 0, w, h)
        self.width  = w
        self.height = h

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindTexture(GL_TEXTURE_2D, self.frame_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, self.copy.data_as_string());
        glBegin(GL_POLYGON);
        glTexCoord2f(0.0, 0.0); glVertex2f(       0.0,         0.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(self.width,         0.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(self.width, self.height)
        glTexCoord2f(0.0, 1.0); glVertex2f(       0.0, self.height)
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0)

        for modelview, type in self.decals:
            glClear(GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluPerspective(45.0, self.width/float(self.height), 0.1, 100.0)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glScalef(1.0, -1.0, -1.0)
            glMultMatrixf(modelview)
            glTranslatef(0.5, 0.5, 0.0)

            if type == 0:   # [W,W,W]
                glColor3f(1.0, 0.1, 0.1)
                glutSolidCone(0.5, 1.0, 100, 20)
            elif type == 4: # [W,B,W]
                glColor3f(0.1, 1.0, 0.1)
                glTranslatef(0.0, 0.0, 0.5)
                glutSolidCube(1.0)
            elif type == 21: # [B,B,B]
                glColor3f(0.1, 0.1, 1.0)
                glTranslatef(0.0, 0.0, 0.1)
                glutSolidTorus(0.2, 0.5, 20, 100)
            elif type == 63: # [R,R,R]
                glColor3f(0.8, 0.8, 0.1)
                glRotatef(90, 1.0, 0.0, 0.0)
                glutSolidTeapot(1.0)
            elif type == 53: # [R,B,B]
                glColor3f(0.1, 0.8, 0.8)
                glTranslatef(0.0, 0.0, 0.5)
                glutSolidSphere(0.5, 50, 50)
            else:
                print "UNKNOWN DECAL:", type

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()

            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

        glutSwapBuffers()

    def on_key(self, k, *args):
        if k in ['q', chr(27)]:
            sys.exit(0)

    def on_idle(self):
        self.frame = cvQueryFrame(self.capture)

        cvFlip(self.frame, self.copy, 1)
        cvCvtColor(self.copy, self.gray, CV_BGR2GRAY)
        cvCanny(self.gray, self.edges, 805, 415, 5) # hand tuned w/ canny.py
        cvDilate(self.edges, self.edges, iterations=1)

        found_decals = []

        for i, decal in enumerate(decals(list(polys(contours(self.edges, self.storage))))):
            ps = [CvPoint2D32f(p.x, p.y) for p in decal.asarray(CvPoint)]
            ps = cvFindCornerSubPix(self.gray, ps, CvSize(5, 5), CvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.001))

            decal_value, orient = self.identify_decal(i, ps, decal)

            # ps = ps[orient:] + ps[:orient] # apply orientation
            for j, p in enumerate(ps):
                self.image_mat[j,0], self.image_mat[j,1] = p.x, p.y

            cvDrawContours(self.copy, decal, CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 2, 8)

            cvFindExtrinsicCameraParams2(self.decal_mat, self.image_mat, self.intrinsic, self.distortion, self.rotation, self.translation)
            cvRodrigues2(self.rotation, self.rotation_matrix)

            modelview = [0.0] * 16
            for x in range(0, 3):
                for y in range(0, 3):
                    modelview[(y*4)+x] = self.rotation_matrix[x,y]
            modelview[12] = self.translation[0,0];
            modelview[13] = self.translation[0,1];
            modelview[14] = self.translation[0,2];
            modelview[15] = 1.0;

            found_decals.append((modelview, decal_value))

        if len(found_decals) > 0:
            self.decals = found_decals

        glutPostRedisplay()

    def identify_decal(self, offset, ps, decal):
        # warp decal back to a flat, 100x100 square
        b = cvBoundingRect(decal, 0)
        cvRectangle(self.copy, cvPoint(b.x,b.y), cvPoint(b.x+b.width, b.y+b.height), CV_RGB(255,0,255), 2, 8, 0)
        cvGetPerspectiveTransform(ps, as_c_array([CvPoint2D32f(0,0), CvPoint2D32f(0,100), CvPoint2D32f(100,100), CvPoint2D32f(100,0)], None, CvPoint2D32f), self.perspective)
        cvWarpPerspective(self.copy, self.temp, self.perspective, CV_WARP_FILL_OUTLIERS)

        # average each of the 4 quadrants
        c = [None] * 4
        cvSetImageROI(self.temp, cvRect(25, 25, 25, 25)); c[0] = cvAvg(self.temp)
        cvSetImageROI(self.temp, cvRect(50, 25, 25, 25)); c[1] = cvAvg(self.temp)
        cvSetImageROI(self.temp, cvRect(50, 50, 25, 25)); c[2] = cvAvg(self.temp)
        cvSetImageROI(self.temp, cvRect(25, 50, 25, 25)); c[3] = cvAvg(self.temp)

        # reorient to put darkest quadrant in top-left (first position)
        c_avg = [None] * 4
        for i in range(0, 4):
            c_avg[i] = (c[i].val[0] + c[i].val[1] + c[i].val[2]) / 3
        orientation = c_avg.index(min(c_avg))
        c = c[orientation:] + c[:orientation]

        # render quadrants to self.copy for debugging purposes
        cvSetImageROI(self.copy, cvRect(offset*100,    0,    50, 50)); cvSet(self.copy, c[0])
        cvSetImageROI(self.copy, cvRect(offset*100+50, 0,    50, 50)); cvSet(self.copy, c[1])
        cvSetImageROI(self.copy, cvRect(offset*100+50, 0+50, 50, 50)); cvSet(self.copy, c[2])
        cvSetImageROI(self.copy, cvRect(offset*100,    0+50, 50, 50)); cvSet(self.copy, c[3])

        # determine color of quadrants, and therefore decal value
        decal_value = 0
        for i in range(1, 4):
            channels = [c[i].val[j] for j in range(0, 3)]
            mx = max(channels)
            mn = min(channels)
            max_channel = channels.index(mx)
            spread = (mx - mn) / float(mx)
            value = (max_channel+1) if spread >= 0.4 else 0
            decal_value = (decal_value * 4) + value
            c[i].val[(max_channel+0)%3] = 0 if value == 0 else 255
            c[i].val[(max_channel+1)%3] = 0
            c[i].val[(max_channel+2)%3] = 0

        # render quadrants to self.copy for debugging purposes
        cvSetImageROI(self.copy, cvRect(offset*100,    100,    50, 50)); cvSet(self.copy, CV_RGB(0,0,0))
        cvSetImageROI(self.copy, cvRect(offset*100+50, 100,    50, 50)); cvSet(self.copy, c[1])
        cvSetImageROI(self.copy, cvRect(offset*100+50, 100+50, 50, 50)); cvSet(self.copy, c[2])
        cvSetImageROI(self.copy, cvRect(offset*100,    100+50, 50, 50)); cvSet(self.copy, c[3])

        cvResetImageROI(self.copy)
        cvResetImageROI(self.temp)

        return (decal_value, orientation)

    def main(self):
        glutMainLoop()

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
        if hole and (contour.rect.width*contour.rect.height) > 400:
            poly = cvApproxPoly(contour, sizeof(CvContour), None, CV_POLY_APPROX_DP, max(contour.rect.width,contour.rect.height)/8)
            if cvCheckContourConvexity(poly):
                yield poly

def decals(polys):
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

if __name__ == '__main__':
    AugmentedReality().main()
