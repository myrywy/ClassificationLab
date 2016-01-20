#ifndef TREEHOGCLASSIFIER_HPP
#define TREEHOGCLASSIFIER_HPP
#include "utils.hpp"
#include "essentials.hpp"
#include "common.hpp"
#include "classifier.h"

#include <QDir>
#include <QFile>
using namespace cv;

class TreeHogClassifier : public Classifier
{
public:
    TreeHogClassifier();
    TreeHogClassifier(QString modelfile);
    virtual ~TreeHogClassifier();
    virtual bool predict(cv::Mat img);
    virtual bool loadFromFile(QString filename);
private:
    CvGBTrees* trees;
    cv::HOGDescriptor* descriptor;
};

#endif // TREEHOGCLASSIFIER_HPP
