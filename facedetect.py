#!/usr/bin/python

from CVtypes import *

scale = 1.3

def main():
    cascade = cv.LoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml', cv.Size(1,1))
    storage = cvCreateMemStorage(0)
    capture = cvCreateCameraCapture(0)
    frame   = cvQueryFrame(capture)
    size    = cvGetSize(frame)
    # roi     = cvCloneImage(frame)
    gray    = cvCreateImage(size, 8, 1)
    small   = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 8, 1)
    eigs    = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 32, 1)
    temp    = cvCreateImage(CvSize(int(size.width/scale), int(size.height/scale)), 32, 1)
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  640)
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480)

    cvNamedWindow('frame', 1)
    # cvNamedWindow('roi',   1)

    state = 'find_face'
    while True:
        if state == 'find_face':
            # cvCopy(frame, roi)
            cvCvtColor(frame, gray, CV_BGR2GRAY)
            cvResize(gray, small, CV_INTER_LINEAR)
            cvEqualizeHist(small, small)
            cvClearMemStorage(storage)

            faces = cvHaarDetectObjects(small, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, CvSize(30, 30))
            if not faces: continue

            r = faces[0]
            tl = CvPoint(int(r.x*scale), int(r.y*scale))
            br = CvPoint(int((r.x+r.width)*scale), int((r.y+r.height)*scale))
            # cvSetImageROI(roi, CvRect(tl.x, tl.y, br.x-tl.x, br.y-tl.y))
            cvRectangle(frame, tl, br, CV_RGB(255, 0, 0), 3, 8, 0)

        cvShowImage('frame', frame)
        # cvShowImage('roi',   roi)

        if cvWaitKey(10) == 0x1B:
            break

        frame = cvQueryFrame(capture)

    cvReleaseImage(gray)
    cvReleaseImage(small)
    cvReleaseCapture(capture)
    cvDestroyWindow('result')

if __name__ == '__main__':
    main()
