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
        self.capture    = cvCaptureFromCAM(0)
        self.frame      = cvQueryFrame(self.capture)
        self.gray       = cvCreateImage(cvSize(self.frame.width, self.frame.height),  8, 1)
        self.mapx       = cvCreateImage(cvSize(self.frame.width, self.frame.height), 32, 1)
        self.mapy       = cvCreateImage(cvSize(self.frame.width, self.frame.height), 32, 1)
        self.intrinsic  = cvCreateMat(3, 3, CV_32FC1)
        self.distortion = cvCreateMat(1, 4, CV_32FC1)

        # from output of calibration.py
        ci = [682.80694580078125, 0.0, 331.3616943359375, 0.0, 631.85980224609375, 210.08140563964844, 0.0, 0.0, 1.0]
        cd = [0.34955242276191711, -0.70636618137359619, -0.013230122625827789, 0.0091487327590584755]

        for x in range(0, 3):
            for y in range(0, 3):
                self.intrinsic[x, y] = ci[(x*3)+y]

        for x in range(0, 4):
            self.distortion[0, x] = cd[x]

        cvInitUndistortMap(self.intrinsic, self.distortion, self.mapx, self.mapy)

    def init_tracker(self):
        self.points = [CvPoint3D32f(x, y, 0) for x in range(0, 3) for y in range(0, 4)]
        self.state  = 'track'
        self.found  = False

        self.chess_mat          = cvCreateMat(12, 3, CV_32FC1)
        self.image_mat          = cvCreateMat(12, 2, CV_32FC1)
        self.rotation           = cvCreateMat(1, 3, CV_32FC1)
        self.rotation_matrix    = cvCreateMat(3, 3, CV_32FC1)
        self.gl_rotation_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self.translation        = cvCreateMat(1, 3, CV_32FC1)

        for i in range(0, 12):
            self.chess_mat[i,0] = self.points[i].x
            self.chess_mat[i,1] = self.points[i].y
            self.chess_mat[i,2] = self.points[i].z

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, self.frame.data_as_string());
        glBegin(GL_POLYGON);
        glTexCoord2f(1.0, 0.0); glVertex2f(  0.0,   0.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(640.0,   0.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(640.0, 480.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(  0.0, 480.0)
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0)

        if self.found:
            glClear(GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            gluPerspective(45.0, self.width/float(self.height), 0.1, 100.0)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            glTranslatef(-self.translation[0,0], -self.translation[0,1], -self.translation[0,2])
            glMultMatrixf(self.gl_rotation_matrix)
            glRotatef(-90.0, 1.0, 0.0, 0.0)
            glTranslatef(-1.5, 1.5, -1.5)

            glFrontFace(GL_CW)
            glColor3f(1.0, 1.0, 1.0)
            glutWireTeapot(2.0)
            glFrontFace(GL_CCW)

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

        cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
        cvEqualizeHist(self.gray, self.gray)

        if self.state == 'track':
            self.state_track()

        glutPostRedisplay()

    def state_track(self):
        self.found, corners = cvFindChessboardCorners(self.gray, CvSize(4, 3), flags=CV_CALIB_CB_ADAPTIVE_THRESH)
        cvDrawChessboardCorners(self.frame, CvSize(4, 3), corners, self.found)

        if self.found:
            for i in range(0, 12):
                self.image_mat[i,0] = corners[i].x
                self.image_mat[i,1] = corners[i].y

            cvFindExtrinsicCameraParams2(self.chess_mat, self.image_mat, self.intrinsic, self.distortion, self.rotation, self.translation)
            cvRodrigues2(self.rotation, self.rotation_matrix)
            for x in range(0,3):
                for y in range(0,3):
                    self.gl_rotation_matrix[(y*4)+x] = self.rotation_matrix[x,y]

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    FaceTracking().main()
