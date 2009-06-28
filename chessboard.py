#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
from random      import *
import sys

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
        self.gray    = cvCreateImage(cvSize(self.frame.width, self.frame.height), 8, 1)
        self.prev    = cvCreateImage(cvSize(self.frame.width, self.frame.height), 8, 1)

    def init_tracker(self):
        self.state = 'find_checkerboard'

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

        cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
        cvEqualizeHist(self.gray, self.gray)

        if self.state == 'find_checkerboard':
            self.state_find_checkerboard()

        cvCopy(self.gray, self.prev)

        glutPostRedisplay()

    def state_find_checkerboard(self):
        corners = cvFindChessboardCorners(self.gray, CvSize(8, 8), 0)
        print "found corners:", len(corners)
        for corner in corners:
            cvCircle(self.frame, cvPoint(int(corner.x), int(corner.y)), 3, CV_RGB(0, 0, 255), 1)

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    FaceTracking().main()
