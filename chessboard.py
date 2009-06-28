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
        self.init_scene()

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

    def init_tracker(self):
        self.points = [CvPoint3D32f(x, y, 0) for x in range(0, 4) for y in range(0, 4)]
        self.state  = 'find_checkerboard'
        self.posit  = cvCreatePOSITObject(self.points)

    def init_scene(self):
        glNewList(self.scene, GL_COMPILE)
        glFrontFace(GL_CW)
        glutSolidTeapot(1.0)
        glFrontFace(GL_CCW)
        glEndList()

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

        if self.state == 'find_checkerboard':
            self.state_find_checkerboard()

        glutPostRedisplay()

    def state_find_checkerboard(self):
        found, corners = cvFindChessboardCorners(self.frame, CvSize(4, 4), flags=CV_CALIB_CB_NORMALIZE_IMAGE)
        cvDrawChessboardCorners(self.frame, CvSize(4, 4), corners, found)
        if found:
            corners = as_c_array([CvPoint2D32f(c.x, c.y) for c in corners], elem_ctype=CvPoint2D32f)
            rot     = as_c_array([0.0]*9, elem_ctype=ctypes.c_float)
            trans   = as_c_array([0.0]*3, elem_ctype=ctypes.c_float)
            cvPOSIT(self.posit, corners, 100.0, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 200, 0.01), rot, trans)
            print [rot[i] for i in range(0, 9)]
            print [trans[i] for i in range(0, 3)]

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    FaceTracking().main()
