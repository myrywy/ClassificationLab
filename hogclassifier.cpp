#include "hogclassifier.hpp"
#include <QDebug>

HogClassifier::HogClassifier()
{
    svm=new cv::SVM;
    params = new cv::SVMParams;
    params->svm_type    = CvSVM::C_SVC;
    params->kernel_type = CvSVM::LINEAR;
    params->term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    descriptor=new cv::HOGDescriptor;
    descriptor->winSize=cv::Size(160,96);
}

HogClassifier::HogClassifier(QString modelFilename)
    : HogClassifier()
{
    loadFromFile(modelFilename);
}

HogClassifier::~HogClassifier()
{
    delete svm;
    delete descriptor;
}

bool HogClassifier::predict(cv::Mat img)
{
    qDebug() << "Klasyfikacja";
    std::vector<float> descriptorValue;
    descriptor->compute(img,descriptorValue);
    qDebug() << "Deskryptor ok";
    int inCols=descriptorValue.size();
    cv::Mat testInput(1,inCols,CV_32FC1);
    for(int i=0;i<inCols;i++){
        testInput.at<float>(0,i)=descriptorValue[i];
    }
    qDebug() << "SVM";
    float fRes=svm->predict(testInput);
    qDebug() << "Klasa: " << fRes;
    if(fRes > 0.1){
        return true;
    }else{
        return false;
    }
}

bool HogClassifier::loadFromFile(QString filename)
{
    qDebug() << "Ladowanie modelu z ";
    qDebug() << filename;
    //filename="/home/marcin/build-ClassificationLab-Desktop-Debug/svm_model.xml";
    svm->load(filename.toStdString().c_str());
    qDebug() << "ok";
}
