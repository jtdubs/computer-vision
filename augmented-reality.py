#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
from random      import *
from ctypes      import *
import sys

from models.laptop import draw_body, draw_lid

from utils.decals import DecalIdentifier

class AugmentedReality:
    def __init__(self):
        self.init_glut()
        self.init_cv()
        self.init_models()
        self.decals = []

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
        self.identifier = DecalIdentifier()

    def init_models(self):
        self.base, self.lid = glGenLists(1), glGenLists(1)

        glNewList(self.base, GL_COMPILE)
        draw_body()
        glEndList()

        glNewList(self.lid, GL_COMPILE)
        draw_lid()
        glEndList()

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

        for modelview, value in self.decals:
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

            if value == 0:   # [W,W,W]
                glColor3f(0.1, 0.1, 0.1)
                glScalef(0.2, 0.2, 0.2)
                glTranslatef(0.0, 4.25, 0.0)
                glCallList(self.base)
                glRotatef(-30, 1, 0, 0)
                glCallList(self.lid)
            elif value == 4: # [W,B,W]
                glColor3f(0.1, 1.0, 0.1)
                glTranslatef(0.0, 0.0, 0.5)
                glutSolidCube(1.0)
            elif value == 21: # [B,B,B]
                glColor3f(0.1, 0.1, 1.0)
                glTranslatef(0.0, 0.0, 0.1)
                glutSolidTorus(0.2, 0.5, 20, 100)
            elif value == 63: # [R,R,R]
                glColor3f(0.8, 0.8, 0.1)
                glTranslatef(0.0, 0.0, 0.25)
                glRotatef(90, 1.0, 0.0, 0.0)
                glutSolidTeapot(0.5)
            elif value == 53: # [R,B,B]
                glColor3f(0.1, 0.8, 0.8)
                glTranslatef(0.0, 0.0, 0.5)
                glutSolidSphere(0.5, 50, 50)
            else:
                print "UNKNOWN DECAL:", value

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

        self.decals = list(self.identifier.get_decals(self.copy))

        glutPostRedisplay()

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    AugmentedReality().main()
