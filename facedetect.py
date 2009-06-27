#!/usr/bin/python

from OpenGL.GLUT import *
from OpenGL.GLU  import *
from OpenGL.GL   import *
from opencv      import *
from math        import *
import sys

class FaceTracking:
    def __init__(self):
        self.scale = 1.3
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
        glutReshapeFunc(self.on_reshape)
        glutDisplayFunc(self.on_display)

    def init_cv(self):
        self.cascade  = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cvSize(1, 1))
        self.storage  = cvCreateMemStorage(0)
        self.capture  = cvCaptureFromCAM(0)
        frame         = cvQueryFrame(self.capture)
        self.gray     = cvCreateImage(cvSize(frame.width, frame.height), 8, 1)
        self.prev     = cvCreateImage(cvSize(frame.width, frame.height), 8, 1)
        self.small    = cvCreateImage(CvSize(int(frame.width/self.scale), int(frame.height/self.scale)),  8, 1)
        self.eigs     = cvCreateImage(CvSize(frame.width, frame.height), 32, 1)
        self.temp     = cvCreateImage(CvSize(frame.width, frame.height), 32, 1)
        self.pyr_a    = cvCreateImage(CvSize(frame.width, frame.height), 32, 1)
        self.pyr_b    = cvCreateImage(CvSize(frame.width, frame.height), 32, 1)
        self.features = None
        cvNamedWindow('frame', 1)

    def init_tracker(self):
        self.state = 'find_face'
        self.flags, self.x, self.y, self.spread, self.distance = 0, 0, 0, 0, 0

    def on_reshape(self, w, h):
        glViewport(0, 0, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w/float(h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        gluLookAt(self.x*2, self.y*2, abs(self.distance*6), 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        for (x, y) in [(0, 0), (3, 1), (-1, -2), (-3, 3), (2, -3)]:
            glColor3f(0.0, 0.0, 1.0); glVertex3f( 0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f( 0.1+x,  0.1+y,  2.0)
            glColor3f(0.0, 1.0, 0.0); glVertex3f( 0.1+x, -0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f( 0.1+x, -0.1+y, -2.0)
            glColor3f(0.0, 1.0, 1.0); glVertex3f( 0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x, -0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y,  2.0)
            glColor3f(1.0, 0.0, 0.0); glVertex3f( 0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f( 0.1+x,  0.1+y, -2.0)
            glColor3f(1.0, 0.0, 1.0); glVertex3f(-0.1+x,  0.1+y,  2.0); glVertex3f(-0.1+x,  0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y, -2.0); glVertex3f(-0.1+x, -0.1+y,  2.0)
            glColor3f(1.0, 1.0, 0.0); glVertex3f( 0.1+x,  0.1+y, -2.0); glVertex3f( 0.1+x,  0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y,  2.0); glVertex3f( 0.1+x, -0.1+y, -2.0)
        glEnd()

        glutSwapBuffers()

    def state_find_face(self, frame):
        best = None

        for face in cvHaarDetectObjects(self.small, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30)):
            r = CvRect(int(face.x*self.scale), int(face.y*self.scale), int(face.width*self.scale), int(face.height*self.scale))
            if not best or r.height > best.height:
                best = r

        if best:
            for image in [self.gray, self.eigs, self.temp]:
                cvSetImageROI(image, best)

            self.features = [x for x in cvGoodFeaturesToTrack(self.gray, self.eigs, self.temp, None, 20, 0.02, 4.0, use_harris=False)]
            min_x, max_x  = 1000, 0
            for f in self.features:
                f.x, f.y = f.x + best.x, f.y + best.y
                min_x, max_x = min(min_x, f.x), max(max_x, f.x)

            for image in [self.gray, self.eigs, self.temp]:
                cvResetImageROI(image)

            anglePerPixel = (3.14159 / 4.5) / 480.0
            angle         = best.width * anglePerPixel

            self.distance = (0.12/2.0) / tan(angle/2.0)
            self.x        = (320 - (best.x + (best.width  / 2.0))) / 160.0 * self.distance
            self.y        = (240 - (best.y + (best.height / 2.0))) / 120.0 * self.distance
            self.spread   = max_x - min_x
            self.flags    = 0
            self.state    = 'track_face'

    def state_track_face(self, frame):
        features, status = cvCalcOpticalFlowPyrLK(self.prev, self.gray, self.pyr_a, self.pyr_b, self.features, None, None, CvSize(50, 50), 3, None, None, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03), self.flags)
        features = [x for x in features]
        dx, dy, min_x, max_x, = 0, 0, 1000, 0
        for i in range(0, len(features)):
            if ord(status[i]) == 0:
                features[i] = None
            else:
                dx = dx + (self.features[i].x - features[i].x)
                dy = dy + (self.features[i].y - features[i].y)
                min_x, max_x = min(min_x, features[i].x), max(max_x, features[i].x)
                cvCircle(frame, cvPoint(int(features[i].x), int(features[i].y)), 3, CV_RGB(0, 0, 255), 1)

        features = [x for x in features if x]
        if len(features) > 0:
            dx, dy = dx / len(features), dy / len(features)

        spread = max_x - min_x

        self.features = features
        self.distance  = self.distance * (self.spread / spread)
        self.x         = self.x + (dx / 160.0 * self.distance)
        self.y         = self.y + (dy / 120.0 * self.distance)
        self.flags     = CV_LKFLOW_PYR_A_READY
        self.spread    = spread

        if len(self.features) < 5:
            self.state = 'find_face'

    def main(self):
        while True:
            frame = cvQueryFrame(self.capture)

            cvCvtColor(frame, self.gray, CV_BGR2GRAY)
            cvResize(self.gray, self.small, CV_INTER_LINEAR)
            cvEqualizeHist(self.small, self.small)
            cvClearMemStorage(self.storage)

            if self.state == 'find_face':
                self.state_find_face(frame)
            elif self.state == 'track_face':
                self.state_track_face(frame)

            cvCopy(self.gray, self.prev)
            self.pyr_a, self.pyr_b = self.pyr_b, self.pyr_a

            cvShowImage('frame', frame)

            if cvWaitKey(10) == 0x1B:
                break

            glutPostRedisplay()
            glutMainLoopEvent()

if __name__ == '__main__':
    FaceTracking().main()
