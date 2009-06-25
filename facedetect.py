#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

from opencv         import *
from opencv.highgui import *

import sys

scale = 1.3

class FaceTracking:
    def __init__(self):
        self.initGlut()
        self.initCV()
        self.initTracker()

    def initGlut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(640,480)
        glutCreateWindow('Face Recognition')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glShadeModel(GL_SMOOTH)
        glutDisplayFunc(self.on_display)

    def initCV(self):
        self.cascade = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cvSize(1, 1))
        self.storage = cvCreateMemStorage(0)
        self.capture = cvCaptureFromCAM(0)
        self.frame   = cvQueryFrame(self.capture)
        self.size    = cvSize(self.frame.width, self.frame.height)
        self.gray    = cvCreateImage(self.size, 8, 1)
        self.small   = cvCreateImage(CvSize(int(self.size.width/scale), int(self.size.height/scale)), 8, 1)
        self.eigs    = cvCreateImage(CvSize(int(self.size.width/scale), int(self.size.height/scale)), 32, 1)
        self.temp    = cvCreateImage(CvSize(int(self.size.width/scale), int(self.size.height/scale)), 32, 1)

        cvNamedWindow('frame', 1)
        cvNamedWindow('small', 1)

    def initTracker(self):
        self.state   = 'find_face'
        self.history = (None, None, None, None, None)

    def cleanupCV(self):
        cvDestroyWindow('result')
        cvReleaseCapture(self.capture)
        cvReleaseImage(self.gray)
        cvReleaseImage(self.small)

    def main(self):
        while True:
            self.frame = cvQueryFrame(self.capture)

            if self.state == 'find_face':
                cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
                cvResize(self.gray, self.small, CV_INTER_LINEAR)
                cvEqualizeHist(self.small, self.small)
                cvClearMemStorage(self.storage)

                faces = cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30))
                if faces:
                    r = faces[0]
                    tl = cvPoint(int(r.x*scale), int(r.y*scale))
                    br = cvPoint(int((r.x+r.width)*scale), int((r.y+r.height)*scale))
                    cvRectangle(self.frame, tl, br, CV_RGB(255, 0, 0), 3, 8, 0)
                    self.history = self.history[1:] + (r,)

                cvShowImage('small', self.small)

            cvShowImage('frame', self.frame)

            if cvWaitKey(10) == 0x1B:
                break

            glutPostRedisplay()
            glutMainLoopEvent()

        self.cleanupCV()

    def on_display(self):
        x, y, s, c = 0, 0, 0, 0
        for r in self.history:
            if r:
                x = x + r.x + (r.width/2)
                y = y + r.y + (r.height/2)
                s = s + ((r.height + r.width)/2)
                c = c + 1
        x, y, s = x / c, y / c, s / c
        print x, y, s
        glClear(GL_COLOR_BUFFER_BIT)
        glutSwapBuffers()

def main():
    FaceTracking().main()

if __name__ == '__main__':
    main()
