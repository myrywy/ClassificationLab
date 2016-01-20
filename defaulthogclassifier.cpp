#include "defaulthogclassifier.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <QDebug>

DefaultHogClassifier::DefaultHogClassifier()
{
    descriptor=new cv::HOGDescriptor;
    descriptor->setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

}

DefaultHogClassifier::~DefaultHogClassifier()
{
    delete descriptor;
}

bool DefaultHogClassifier::predict(cv::Mat img)
{
    cv::Mat img90, img270, rotationMat90, rotationMat270;
    std::vector<cv::Rect> found90, found270;
    cv::Size rotatedSize(img.size().height,img.size().width);
    cv::Point center(img.size().width/2, img.size().height/2);
    //Macierz translacji, żeby się mieścił obrócony obraz w nowej macierzy
    double t90[3][3]={
        {1.0,0.0,-img.size().width/2.0+img.size().height/2.0},
        {0.0,1.0,+img.size().height/2.0-img.size().width/2.0},
        {0.0,0.0,1.0}
    };
    cv::Mat translationMat90(3,3,CV_64FC1,t90);
    double t270[3][3]={
                {1,0,-img.size().height/2.0+img.size().width/2.0},
                {0,1,+img.size().width/2.0-img.size().height/2.0},
                {0,0,1}
            };
    cv::Mat translationMat270(3,3,CV_64FC1,t270);
    rotationMat90 = cv::getRotationMatrix2D( center, 90, 1 )*translationMat90;
    rotationMat270 = cv::getRotationMatrix2D( center, 270, 1 )*translationMat270;
    cv::warpAffine( img, img90, rotationMat90, rotatedSize );
    cv::warpAffine( img, img270, rotationMat270, rotatedSize );

    cv::imshow("90",img90);
    cv::imshow("270",img270);
    cv::waitKey(2000);

    descriptor->detectMultiScale(img90,found90, 0 , cv::Size(8,8), cv::Size(32,32),1.05,2);
    descriptor->detectMultiScale(img270,found270, 0 , cv::Size(8,8), cv::Size(32,32),1.05,2);
    qDebug() << "f90 " << found90.size() << " f270 " << found270.size();
    if(!(found90.empty() && found270.empty())){
        return true;
    }else{
        return false;
    }
}
