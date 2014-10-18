/*************************************************************************
> File Name: exampleCanny.cpp
> Author: yy
> Mail: mengyy_linux@163.com
 ************************************************************************/

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace std;

void usage(void)
{
    cout << "./exampleCanny fileName" <<endl;
}

IplImage* doCanny(IplImage *in, double lowThresh, double highThresh, double aperture)
{
    IplImage *out = NULL;

    out = cvCreateImage(cvGetSize(in), IPL_DEPTH_8U, 1);
    if (out == NULL)
    {
        cerr << "cvCreateImage error." << endl;
        return NULL;
    }

    cvCanny(in, out, lowThresh, highThresh, aperture);
    return out;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "argv error." << endl;
        usage();
        return -1;
    }

    const char *windowName = "canny";
    const char *fileName = argv[1];
    IplImage *img = NULL;
    IplImage *gray = NULL;

    cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
    img = cvLoadImage(fileName);

    gray = doCanny(img, 10, 100, 3);
    cvShowImage(windowName, gray);
    cvWaitKey(0);

    cvReleaseImage(&gray);
    cvReleaseImage(&img);
    cvDestroyWindow(windowName);

    return 0;
}
