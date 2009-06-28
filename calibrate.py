#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
from random      import *
import sys
import ctypes

class FaceTracking:
    def __init__(self):
        self.init_glut()
        self.init_cv()
        self.init_tracker()

    def init_glut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutCreateWindow('Face Tracking')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glutReshapeFunc(self.on_reshape)
        glutDisplayFunc(self.on_display)
        glutKeyboardFunc(self.on_key)
        glutIdleFunc(self.on_idle)

        self.frame_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.frame_texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        self.scene = glGenLists(1)

    def init_cv(self):
        self.capture = cvCaptureFromCAM(0)
        self.frame   = cvQueryFrame(self.capture)
        self.gray    = cvCreateImage(cvSize(self.frame.width, self.frame.height), 8, 1)

    def init_tracker(self):
        self.points = [CvPoint3D32f(x, y, 0) for x in range(0, 3) for y in range(0, 4)]
        self.state  = 'calibrate'

        self.chess_mat = cvCreateMat(8*12, 3, CV_32FC1)
        for n in range(0, 8):
            for i in range(0, 12):
                self.chess_mat[(n*12)+i,0] = self.points[i].x
                self.chess_mat[(n*12)+i,1] = self.points[i].y
                self.chess_mat[(n*12)+i,2] = self.points[i].z

        self.image_mat  = cvCreateMat(8*12, 2, CV_32FC1)
        self.counts     = cvCreateMat(8,    1, CV_32SC1)
        self.intrinsic  = cvCreateMat(3,    3, CV_32FC1)
        self.distortion = cvCreateMat(1,    4, CV_32FC1)
        self.n          = 0

    def on_reshape(self, w, h):
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, self.frame.data_as_string());
        glBegin(GL_POLYGON);
        glTexCoord2f(1.0, 0.0); glVertex2f(0.0,        0.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(self.width, 0.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(self.width, self.height)
        glTexCoord2f(1.0, 1.0); glVertex2f(0.0,        self.height)
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0)

        glutSwapBuffers()

    def on_key(self, k, *args):
        if k in ['q', chr(27)]:
            sys.exit(0)

    def on_idle(self):
        self.frame = cvQueryFrame(self.capture)

        cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
        cvEqualizeHist(self.gray, self.gray)

        if self.state == 'calibrate':
            self.state_calibrate()
        elif self.state == 'delay':
            self.state_delay()
        elif self.state == 'display':
            self.state_display()

        glutPostRedisplay()

    def state_calibrate(self):
        found, corners = cvFindChessboardCorners(self.gray, CvSize(4, 3), flags=CV_CALIB_CB_ADAPTIVE_THRESH)
        cvDrawChessboardCorners(self.frame, CvSize(4, 3), corners, found)

        if found:
            corners = cvFindCornerSubPix(self.gray, corners, CvSize(5, 5), CvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.01))

            for i in range(0, 12):
                self.image_mat[(self.n*12)+i,0] = corners[i].x
                self.image_mat[(self.n*12)+i,1] = corners[i].y
                self.counts[self.n,0]           = len(corners)

            self.n = self.n + 1

            if self.n == 8:
                self.state = 'display'
            else:
                self.skip_frames = 30
                self.state       = 'delay'

    def state_delay(self):
        if self.skip_frames == 0:
            self.state = 'calibrate'
        else:
            self.skip_frames = self.skip_frames - 1

    def state_display(self):
        cvCalibrateCamera2(self.chess_mat, self.image_mat, self.counts, CvSize(640, 480), self.intrinsic, self.distortion, flags=0)

        print "instrinsic =", [self.intrinsic[x,y]  for x in range(0, 3) for y in range(0, 3)]
        print "distortion =", [self.distortion[0,i] for i in range(0, 4)]
        sys.exit(0)

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    FaceTracking().main()
