#ifndef DEFAULTHOGCLASSIFIER_HPP
#define DEFAULTHOGCLASSIFIER_HPP

#include "classifier.h"
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
//#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class DefaultHogClassifier: public Classifier
{
public:
    DefaultHogClassifier();
    virtual ~DefaultHogClassifier();
    virtual bool predict(cv::Mat img);
    virtual bool loadFromFile(QString filename){;}
private:
    cv::HOGDescriptor* descriptor;
};

#endif // DEFAULTHOGCLASSIFIER_HPP
