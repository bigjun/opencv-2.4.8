/*************************************************************************
> File Name: exampleImg.cpp
> Author: yy
> Mail: mengyy_linux@163.com
 ************************************************************************/

#include <opencv/highgui.h>
#include <iostream>
using namespace std;

void usage(void)
{
    cout << "./exampleImg fileName" <<endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "argv error." << endl;
        usage();
        return -1;
    }

    const char *windowName = "result";
    const char *fileName = argv[1];
    IplImage *img = NULL;

    cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
    img = cvLoadImage(fileName);
    cvShowImage(windowName, img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvDestroyWindow(windowName);

    return 0;
}
