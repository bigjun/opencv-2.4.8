/*************************************************************************
> File Name: exampleImg.cpp
> Author: yy
> Mail: mengyy_linux@163.com
 ************************************************************************/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;

void usage(void)
{
    cout << "./exampleSmooth fileName" <<endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "argv error." << endl;
        usage();
        return -1;
    }

    const char *windowName = "smooth";
    const char *fileName = argv[1];
    IplImage *img = NULL;
    IplImage *out = NULL;

    cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
    img = cvLoadImage(fileName);
    out = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

    /* do the smoothing */
    cvSmooth(img, out, CV_GAUSSIAN, 3, 3);

    cvShowImage(windowName, out);
    cvWaitKey(0);

    cvReleaseImage(&out);
    cvReleaseImage(&img);
    cvDestroyWindow(windowName);

    return 0;
}
