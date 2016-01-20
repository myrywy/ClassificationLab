#ifndef HOGCLASSIFIER_HPP
#define HOGCLASSIFIER_HPP

#include "classifier.h"
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
//#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class HogClassifier : public Classifier
{
public:
    HogClassifier();
    HogClassifier(QString modelFilename);
    virtual ~HogClassifier();
    virtual bool predict(cv::Mat img);
    virtual bool loadFromFile(QString filename);
private:
    cv::SVM* svm;
    cv::HOGDescriptor* descriptor;
    cv::SVMParams* params;
};

#endif // HOGCLASSIFIER_HPP
