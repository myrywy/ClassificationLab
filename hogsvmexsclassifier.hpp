#ifndef HOGSVMEXSCLASSIFIER_HPP
#define HOGSVMEXSCLASSIFIER_HPP

#include "classifier.h"
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
//#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "exhaustivesearch.hpp"

class HogSvmExSClassifier : public Classifier
{
public:
    HogSvmExSClassifier();
    HogSvmExSClassifier(QString modelFilename);
    virtual ~HogSvmExSClassifier();
    virtual bool predict(cv::Mat img);
    virtual bool loadFromFile(QString filename);
private:
    cv::SVM* svm;
    cv::HOGDescriptor* descriptor;
    ExhaustiveSearch& exSearch;
};


#endif // HOGSVMEXSCLASSIFIER_HPP
