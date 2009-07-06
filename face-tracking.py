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
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
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
        self.cascade  = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cvSize(1, 1))
        self.storage  = cvCreateMemStorage(0)
        self.capture  = cvCaptureFromCAM(0)
        self.frame    = cvQueryFrame(self.capture)
        self.gray     = cvCreateImage(cvSize(self.frame.width, self.frame.height), 8, 1)
        self.prev     = cvCreateImage(cvSize(self.frame.width, self.frame.height), 8, 1)
        self.eigs     = cvCreateImage(CvSize(self.frame.width, self.frame.height), 32, 1)
        self.temp     = cvCreateImage(CvSize(self.frame.width, self.frame.height), 32, 1)
        self.pyr_a    = cvCreateImage(CvSize(self.frame.width, self.frame.height), 32, 1)
        self.pyr_b    = cvCreateImage(CvSize(self.frame.width, self.frame.height), 32, 1)
        self.features = None

    def init_tracker(self):
        self.state = 'choose_face'
        self.x, self.y, self.distance = 0, 0, 8
        self.show_frame = False

    def init_scene(self):
        def generate_target():
            x, y, z = uniform(-7.5, 7.5), uniform(-4.5, 4.5), uniform(-20, 0)
            return (x, y, z, (z+20.0)/20.0, 0, 0)
        targets = [generate_target() for x in range(0, 50)]

        glNewList(self.scene, GL_COMPILE)

        glBegin(GL_LINES)
        for z in range(-20, 1):
            glColor3f(1.0+(z/20.0), 1.0+(z/20.0), 1.0+(z/20.0))
            glVertex3f(-8, -5, z); glVertex3f(-8,  5, z)
            glVertex3f(-8, -5, z); glVertex3f( 8, -5, z)
            glVertex3f( 8,  5, z); glVertex3f(-8,  5, z)
            glVertex3f( 8,  5, z); glVertex3f( 8, -5, z)
        for x in range(-8, 9):
            glColor3f(1.0, 1.0, 1.0); glVertex3f(x, -5, 0); glColor3f(0.0, 0.0, 0.0); glVertex3f(x, -5, -20)
            glColor3f(1.0, 1.0, 1.0); glVertex3f(x,  5, 0); glColor3f(0.0, 0.0, 0.0); glVertex3f(x,  5, -20)
        for y in range(-5, 6):
            glColor3f(1.0, 1.0, 1.0); glVertex3f(-8, y, 0); glColor3f(0.0, 0.0, 0.0); glVertex3f(-8, y, -20)
            glColor3f(1.0, 1.0, 1.0); glVertex3f( 8, y, 0); glColor3f(0.0, 0.0, 0.0); glVertex3f( 8, y, -20)
        for (x, y, z, r, g, b) in targets:
            glColor3f(1.0, 1.0, 1.0); glVertex3f(x, y, z-0.02); glColor3f(0.0, 0.0, 0.0); glVertex3f(x, y, -20)
        glEnd()

        glBegin(GL_QUADS)
        for (x, y, z, r, g, b) in targets:
            glColor3f(r, g, b); glVertex3f(0.5+x, 0.5+y, z); glVertex3f(-0.5+x, 0.5+y, z); glVertex3f(-0.5+x, -0.5+y, z); glVertex3f(0.5+x, -0.5+y, z)
            glColor3f(r*0.7, g*0.7, b*0.7); glVertex3f(0.6+x, 0.6+y, z-0.01); glVertex3f(-0.6+x, 0.6+y, z-0.01); glVertex3f(-0.6+x, -0.6+y, z-0.01); glVertex3f(0.6+x, -0.6+y, z-0.01)
        glEnd()

        glEndList()

    def on_reshape(self, w, h):
        glViewport(0, 0, w, h)
        self.width  = w
        self.height = h

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w/float(h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, 5.0)
        glTranslatef(-self.x, -self.y, -self.distance)
        glCallList(self.scene);

        if self.show_frame:
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, self.height, 0, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glBindTexture(GL_TEXTURE_2D, self.frame_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, self.frame.data_as_string());
            glBegin(GL_POLYGON);
            glTexCoord2f(1.0, 0.0); glVertex2f(  0.0,   0.0)
            glTexCoord2f(0.0, 0.0); glVertex2f(640.0,   0.0)
            glTexCoord2f(0.0, 1.0); glVertex2f(640.0, 480.0)
            glTexCoord2f(1.0, 1.0); glVertex2f(  0.0, 480.0)
            glEnd();
            glBindTexture(GL_TEXTURE_2D, 0)

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()

            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

        glutSwapBuffers()

    def on_key(self, k, *args):
        if   k in ['q', chr(27)]:          sys.exit(0)
        elif k in ['m', chr(10), chr(13)]: self.state = 'mark_face'
        elif k in ['c']:                   self.state = 'choose_face'
        elif k in ['s']:                   self.init_scene()
        elif k in ['f']:                   self.show_frame = not self.show_frame

    def on_idle(self):
        self.frame = cvQueryFrame(self.capture)

        cvCvtColor(self.frame, self.gray, CV_BGR2GRAY)
        cvEqualizeHist(self.gray, self.gray)

        if   self.state == 'choose_face': self.state_choose_face()
        elif self.state == 'mark_face':   self.state_mark_face()
        elif self.state == 'track_face':  self.state_track_face()

        cvCopy(self.gray, self.prev)
        glutPostRedisplay()

    def state_choose_face(self):
        best = None
        cvClearMemStorage(self.storage)
        for face in cvHaarDetectObjects(self.gray, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(100, 100)):
            conservative = CvRect(face.x+30, face.y+20, face.width-60, face.height-40)
            if not best or conservative.height > best.height:
                best = conservative

            cvRectangle(self.frame,
                        CvPoint(conservative.x,                    conservative.y),
                        CvPoint(conservative.x+conservative.width, conservative.y+conservative.height),
                        CV_RGB(255, 0, 0), 3, 8, 0)

        if best:
            cvRectangle(self.frame, CvPoint(best.x, best.y), CvPoint(best.x+best.width, best.y+best.height), CV_RGB(0, 255, 0), 3, 8, 0)

    def state_mark_face(self):
        best = None
        cvClearMemStorage(self.storage)
        for face in cvHaarDetectObjects(self.gray, self.cascade, self.storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(100, 100)):
            conservative = CvRect(face.x+30, face.y+20, face.width-60, face.height-40)
            if not best or conservative.height > best.height:
                best = conservative

        if not best:
            return

        for image in [self.gray, self.eigs, self.temp]:
            cvSetImageROI(image, best)

        features = cvGoodFeaturesToTrack(self.gray, self.eigs, self.temp, None, 100, 0.05, 6.0, use_harris=False)
        cvFindCornerSubPix(self.gray, features, CvSize(5, 5), CvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01))
        self.features = [x for x in features]

        avg_x, avg_y, min_y, max_y  = 0, 0, 1000, 0
        for f in self.features:
            f.x, f.y = f.x + best.x, f.y + best.y
            min_y, max_y = min(min_y, f.y), max(max_y, f.y)
            avg_x, avg_y = avg_x + f.x,     avg_y + f.y
        avg_x, avg_y = avg_x / len(self.features), avg_y / len(self.features)

        for image in [self.gray, self.eigs, self.temp]:
            cvResetImageROI(image)

        self.start_avg_x    = avg_x
        self.start_avg_y    = avg_y
        self.start_x        = self.x        = 0
        self.start_y        = self.y        = 0
        self.start_spread   = max_y - min_y
        self.start_distance = self.distance = 8.0
        self.flags          = 0
        self.state          = 'track_face'

    def state_track_face(self):
        self.pyr_a, self.pyr_b = self.pyr_b, self.pyr_a
        features, status = cvCalcOpticalFlowPyrLK(self.prev, self.gray, self.pyr_a, self.pyr_b, self.features, None, None, CvSize(30, 30), 3, None, None, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.03), self.flags)
        features = [x for x in features]

        avg_x, avg_y, min_y, max_y = 0, 0, 1000, 0
        for i in range(0, len(features)):
            if ord(status[i]) == 0:
                features[i] = None
            else:
                avg_x, avg_y = avg_x + features[i].x,     avg_y + features[i].y
                min_y, max_y = min(min_y, features[i].y), max(max_y, features[i].y)
                cvCircle(self.frame, cvPoint(int(features[i].x), int(features[i].y)), 3, CV_RGB(0, 0, 255), 1)

        features     = [x for x in features if x]
        avg_x, avg_y = avg_x / len(features), avg_y / len(features)
        spread       = max_y - min_y

        if len(features) < 20:
            self.state = 'mark_face'
            return

        self.features = features
        self.x        = self.start_x + ((self.start_avg_x - avg_x) / 320.0 * self.distance)
        self.y        = self.start_y + ((self.start_avg_y - avg_y) / 240.0 * self.distance)
        self.distance = self.start_distance * ((self.start_spread / spread) ** 2)
        self.flags    = CV_LKFLOW_PYR_A_READY

    def main(self):
        glutMainLoop()

if __name__ == '__main__':
    FaceTracking().main()
