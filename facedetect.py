#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

from opencv         import *
from opencv.highgui import *

import sys
import math

class FaceTracking:
    def __init__(self):
        self.window = 5
        self.scale  = 1.3

        self.init_glut()
        self.init_cv()
        self.init_tracker()

    def init_glut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(640,480)
        glutCreateWindow('Face Recognition')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glutReshapeFunc(self.on_reshape)
        glutDisplayFunc(self.on_display)

    def init_cv(self):
        self.cascade = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cvSize(1, 1))
        self.storage = cvCreateMemStorage(0)
        self.capture = cvCaptureFromCAM(0)
        frame        = cvQueryFrame(self.capture)
        self.gray    = cvCreateImage(cvSize(frame.width, frame.height), 8, 1)
        self.small   = cvCreateImage(CvSize(int(frame.width/self.scale), int(frame.height/self.scale)), 8, 1)

        cvNamedWindow('frame', 1)
        cvNamedWindow('small', 1)

    def init_tracker(self):
        self.state   = 'find_face'
        self.history = (None,) * self.window

    def cleanup_cv(self):
        cvDestroyWindow('result')

    def rect_to_params(self, rect):
        anglePerPixel = (3.14159 / 4.5) / 480.0
        angle         = rect.width * anglePerPixel
        distance      = (0.12/2.0) / math.tan(angle/2.0)
        x             = (320 - (rect.x + (rect.width  / 2.0))) / 160.0 * distance
        y             = (240 - (rect.y + (rect.height / 2.0))) / 120.0 * distance
        return (x, y, rect.height, distance)

    def on_reshape(self, w, h):
        print "viewport:", w, h

        glViewport(0, 0, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w/float(h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_display(self):
        x, y, d, c = 0, 0, 0, 0
        for (x1, y1, h1, d1) in [h for h in self.history if h]:
            x, y, d, c = x+x1, y+y1, d+d1, c+1
        if c > 0:
            x, y, d = x / c, y / c, d / c

        print x, y, d

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        gluLookAt(x*2, y*2, abs(d*6), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        for (x, y) in [(0, 0), (3, 1), (-1, -2), (-3, 3), (2, -3)]:
            glColor3f(0.0, 1.0, 0.0); glVertex3f( 0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f( 0.1+x,  0.1+y,  2.0)
            glColor3f(1.0, 0.1, 0.0); glVertex3f( 0.1+x, -0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f( 0.1+x, -0.1+y, -2.0)
            glColor3f(1.0, 0.0, 0.0); glVertex3f( 0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y,  2.0)
            glColor3f(1.0, 1.0, 0.0); glVertex3f( 0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f( 0.1+x,  0.1+y, -2.0)
            glColor3f(0.0, 0.0, 1.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y,  2.0)
            glColor3f(1.0, 0.0, 1.0); glVertex3f( 0.1+x,  0.1+y, -2.0); glVertex3f( 0.1+x,  0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y, -2.0)
        glEnd()

        glutSwapBuffers()

    def main(self):
        while True:
            frame = cvQueryFrame(self.capture)

            cvCvtColor(frame, self.gray, CV_BGR2GRAY)
            cvResize(self.gray, self.small, CV_INTER_LINEAR)
            cvEqualizeHist(self.small, self.small)
            cvClearMemStorage(self.storage)

            if self.state == 'find_face':
                best = None

                for face in cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30)):
                    r = CvRect(int(face.x*self.scale), int(face.y*self.scale), int(face.width*self.scale), int(face.height*self.scale))
                    if not best or r.height > best.height:
                        best = r
                    cvRectangle(frame, CvPoint(r.x, r.y), CvPoint(r.x+r.width, r.y+r.height), CV_RGB(255, 0, 0), 3, 8, 0)

                if best:
                    self.history = self.history[1:] + (self.rect_to_params(best),)
                    self.state = 'track_face'

            elif self.state == 'track_face':
                (x, y, height, size) = self.history[-1]
                best = None

                for face in cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30)):
                    r = CvRect(int(face.x*self.scale), int(face.y*self.scale), int(face.width*self.scale), int(face.height*self.scale))
                    if not best or abs(r.height-height) < abs(best.height-height):
                        best = r
                    cvRectangle(frame, CvPoint(r.x, r.y), CvPoint(r.x+r.width, r.y+r.height), CV_RGB(0, 0, 255), 3, 8, 0)

                if best:
                    cvRectangle(frame, CvPoint(best.x, best.y), CvPoint(best.x+best.width, best.y+best.height), CV_RGB(0, 255, 0), 3, 8, 0)
                    self.history = self.history[1:] + (self.rect_to_params(best),)
                else:
                    self.state = 'find_face'

            cvShowImage('small', self.small)
            cvShowImage('frame', frame)

            if cvWaitKey(10) == 0x1B:
                break

            glutPostRedisplay()
            glutMainLoopEvent()

        self.cleanup_cv()

def main():
    FaceTracking().main()

if __name__ == '__main__':
    main()
