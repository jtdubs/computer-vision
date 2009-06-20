#!/usr/bin/python

from CVtypes import *

scale = 1.3

def main():
    cascade = cv.LoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cv.Size(1,1))
    storage = cvCreateMemStorage(0)
    capture = cvCreateCameraCapture(0)
    frame   = cvQueryFrame(capture)
    size    = cvGetSize(frame)
    gray    = cvCreateImage(size, 8, 1)
    small   = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 8, 1)


    cvNamedWindow('result', 1)

    while True:
        cvCvtColor(frame, gray, CV_BGR2GRAY)
        cvResize(gray, small, CV_INTER_LINEAR)
        cvEqualizeHist(small, small)
        cvClearMemStorage(storage)

        faces = cvHaarDetectObjects(small, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, CvSize(30, 30))
        for r in faces:
            cvRectangle(frame, CvPoint(int(r.x*scale), int(r.y*scale)), CvPoint(int((r.x+r.width)*scale), int((r.y+r.height)*scale)), CV_RGB(255, 0, 0), 3, 8, 0)

        cvShowImage('result', frame)
        if cvWaitKey(10) >= 0: break

        frame = cvQueryFrame(capture)

    cvReleaseImage(gray)
    cvReleaseImage(small)
    cvReleaseCapture(capture)
    cvDestroyWindow('result')

if __name__ == '__main__':
    main()
