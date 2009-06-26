#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *

from opencv         import *
from opencv.highgui import *

import sys
import math

scale  = 1.3
window = 8

class FaceTracking:
    def __init__(self):
        self.init_glut()
        self.init_cv()
        self.init_tracker()

    def init_glut(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
        glutInitWindowSize(640,480)
        glutCreateWindow('Face Recognition')

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glShadeModel(GL_SMOOTH)
        glutDisplayFunc(self.on_display)

    def init_cv(self):
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

    def init_tracker(self):
        self.state   = 'find_face'
        self.history = (None,) * window

    def cleanup_cv(self):
        cvDestroyWindow('result')
        # cvReleaseCapture(self.capture)
        cvReleaseImage(self.gray)
        cvReleaseImage(self.small)

    def rect_to_params(self, rect):
        size     = (rect.width+rect.height)/2
        x        = rect.x + (rect.width/2)
        y        = rect.y + (rect.height/2)
        distance = (110 - math.sqrt(13600 - 40*(340-size))) / 20
        return (x, y, rect.height, distance)

    def on_display(self):
        x, y, d, c = 0, 0, 0, 0
        for (x1, y1, h1, d1) in [h for h in self.history if h]:
            x, y, d, c = x+x1, y+y1, d+d1, c+1
        x, y, d = x / c, y / c, d / c

        print x, y, d
        glClear(GL_COLOR_BUFFER_BIT)
        glutSwapBuffers()

    def main(self):
        while True:
            self.frame = cvQueryFrame(self.capture)

            cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
            cvResize(self.gray, self.small, CV_INTER_LINEAR)
            cvEqualizeHist(self.small, self.small)
            cvClearMemStorage(self.storage)

            if self.state == 'find_face':
                best = None

                for face in cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30)):
                    r = CvRect(int(face.x*scale), int(face.y*scale), int(face.width*scale), int(face.height*scale))
                    if not best or r.height > best.height:
                        best = r
                    cvRectangle(self.frame, CvPoint(r.x, r.y), CvPoint(r.x+r.width, r.y+r.height), CV_RGB(255, 0, 0), 3, 8, 0)

                if best:
                    self.history = self.history[1:] + (self.rect_to_params(best),)
                    self.state = 'track_face'

            elif self.state == 'track_face':
                (x, y, height, size) = self.history[-1]
                best = None

                for face in cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30)):
                    r = CvRect(int(face.x*scale), int(face.y*scale), int(face.width*scale), int(face.height*scale))
                    if not best or abs(r.height-height) < abs(best.height-height):
                        best = r
                    cvRectangle(self.frame, CvPoint(r.x, r.y), CvPoint(r.x+r.width, r.y+r.height), CV_RGB(0, 0, 255), 3, 8, 0)

                if best:
                    cvRectangle(self.frame, CvPoint(best.x, best.y), CvPoint(best.x+best.width, best.y+best.height), CV_RGB(0, 255, 0), 3, 8, 0)
                    self.history = self.history[1:] + (self.rect_to_params(best),)
                else:
                    self.state = 'find_face'

            cvShowImage('small', self.small)
            cvShowImage('frame', self.frame)

            if cvWaitKey(10) == 0x1B:
                break

            glutPostRedisplay()
            glutMainLoopEvent()

        self.cleanup_cv()

def main():
    FaceTracking().main()

if __name__ == '__main__':
    main()
