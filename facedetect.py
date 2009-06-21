#!/usr/bin/python

from opencv         import *
from opencv.highgui import *

scale = 1.3

def main():
    cascade = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cvSize(1, 1))
    storage = cvCreateMemStorage(0)
    capture = cvCaptureFromCAM(0)
    frame   = cvQueryFrame(capture)
    size    = cvSize(frame.width, frame.height)
    gray    = cvCreateImage(size, 8, 1)
    small   = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 8, 1)
    eigs    = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 32, 1)
    temp    = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 32, 1)

    cvNamedWindow('frame', 1)
    cvNamedWindow('small', 1)

    state = 'find_face'
    while True:
        frame = cvQueryFrame(capture)

        if state == 'find_face':
            cvCvtColor(frame, gray, CV_BGR2GRAY)
            cvResize(gray, small, CV_INTER_LINEAR)
            cvEqualizeHist(small, small)
            cvClearMemStorage(storage)

            faces = cvHaarDetectObjects(small, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30))
            if faces:
                r = faces[0]
                tl = cvPoint(int(r.x*scale), int(r.y*scale))
                br = cvPoint(int((r.x+r.width)*scale), int((r.y+r.height)*scale))
                cvRectangle(frame, tl, br, CV_RGB(255, 0, 0), 3, 8, 0)

                for (x, y) in cvGoodFeaturesToTrack(small, eigs, temp, None, 20, 1.0, 1.0, use_harris=True):
                    print (x, y)

        cvShowImage('small', small)
        cvShowImage('frame', frame)

        if cvWaitKey(10) == 0x1B:
            break

    cvReleaseImage(gray)
    cvReleaseImage(small)
    cvReleaseCapture(capture)
    cvDestroyWindow('result')

if __name__ == '__main__':
    main()
