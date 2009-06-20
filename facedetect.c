#include "cv.h"
#include "highgui.h"

const char* cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml";
double scale = 1.3;

int main() {
    CvHaarClassifierCascade *cascade = cvLoad(cascade_name, 0, 0, 0);
    CvMemStorage            *storage = cvCreateMemStorage(0);
    CvCapture               *capture = cvCaptureFromCAM(0);

    cvNamedWindow("result", 1);

    IplImage *frame = cvQueryFrame(capture);
    IplImage *gray  = cvCreateImage(cvSize(frame->width, frame->height), 8, 1);
    IplImage *small = cvCreateImage(cvSize(cvRound(frame->width/scale), cvRound(frame->height/scale)), 8, 1);

    while (1) {
        cvCvtColor(frame, gray, CV_BGR2GRAY);
        cvResize(gray, small, CV_INTER_LINEAR);
        cvEqualizeHist(small, small);
        cvClearMemStorage(storage);

        CvSeq* faces = cvHaarDetectObjects(small, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));
        if (faces) {
            CvRect r = *(CvRect*)cvGetSeqElem(faces, 0);
            cvRectangle(frame, cvPoint(r.x*scale,r.y*scale), cvPoint((r.x+r.width)*scale, (r.y+r.height)*scale), CV_RGB(255,0,0), 3, 8, 0);
        }

        cvShowImage("result", frame);

        if(cvWaitKey(10) >= 0)
            break;

        frame = cvQueryFrame(capture);
    }

    cvReleaseImage(&gray);
    cvReleaseImage(&small);
    cvReleaseCapture(&capture);
    cvDestroyWindow("result");

    return 0;
}
