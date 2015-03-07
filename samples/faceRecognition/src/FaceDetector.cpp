/*************************************************************************
> File Name: facedetector.cpp
> Author: yy
> Mail: mengyy_linux@163.com
 ************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
     
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

class MyFaceDetection
{
public:
    MyFaceDetection(string _cascadeName = "./feature_lib_data/haarcascade_frontalface_alt2.xml");
    ~MyFaceDetection();

    void init(void);
    void detect(Mat &img);
    vector<Rect> getFacesRegion(void) const;

private:
    string cascadeName;
    CascadeClassifier *cascade;
    vector<Rect> facesRegion;
};

MyFaceDetection::MyFaceDetection(string _cascadeName)
:cascadeName(_cascadeName)
{
    init();
}

MyFaceDetection::~MyFaceDetection()
{
    delete cascade;
}

void MyFaceDetection::init(void)
{
    cascade = new CascadeClassifier();
    cascade->load(cascadeName);

    facesRegion.clear();
}

void MyFaceDetection::detect(Mat &img)
{
    Mat gray;

    cvtColor(img, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);

    cascade->detectMultiScale(gray, facesRegion, 1.25, 3, 0, Size(50, 50));
}

vector<Rect> MyFaceDetection::getFacesRegion(void) const
{
    return facesRegion;
}

int main(int argc, char**argv)
{
    if (argc < 2)
    {
        return -1;
    }

    string filename;
    Mat img;
    vector<Rect> faces;
    MyFaceDetection faceDetection;

    filename.assign(argv[1]);

    img = imread(filename, 1);
    if (img.empty())
    {
        cout << "read image " << filename.data() << "error" << endl;
        return -1;
    }

    faceDetection.detect(img);
    faces = faceDetection.getFacesRegion();

    int n = 1;
    for (vector<Rect>::const_iterator it = faces.begin(); it != faces.end(); it++)
    {
        string facename;
        string msg = format("face rectangle region: x:%d y:%d width:%d height:%d ",
                                        it->x , it->y, it->width, it->height);
        
        Mat faceImg(img, Range(it->y, it->y + it->height), Range(it->x, it->x + it->width)); 
        facename = format("face-%d.jpeg", n++);
        imwrite(facename, faceImg);

        cout << msg << endl;
        //rectangle(img, cvPoint(it->x, it->y), cvPoint(it->x + it->width - 1, it->y + it->height - 1),  CV_RGB(255,128,0), 3, 8, 0);
    }

    //imshow("gray image", img);
    //waitKey(0);

    return 0;
}
