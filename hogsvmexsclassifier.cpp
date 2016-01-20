#include "hogsvmexsclassifier.hpp"

HogSvmExSClassifier::HogSvmExSClassifier()
    :exSearch(*(new ExhaustiveSearch))
{
    svm=new cv::SVM;
    descriptor=new HOGDescriptor;
    descriptor->winSize=cv::Size(160,96);
    exSearch.setRatio(double(160)/96);
    exSearch.setStride(0.5);
    exSearch.setScales(QVector<double>({1,0.9}));
    auto detectionF = [&](Mat img)->bool{
        resize(img,img,Size(160,96));
        std::vector<float> descriptorValue;
        descriptor->compute(img,descriptorValue);
        int inCols=descriptorValue.size();
        cv::Mat testInput(1,inCols,CV_32FC1);
        for(int i=0;i<inCols;i++){
            testInput.at<float>(0,i)=descriptorValue[i];
        }
        float fRes=svm->predict(testInput);
        if(fRes > 0.1){
            return true;
        }else{
            return false;
        }
    };
    exSearch.setDetectionFunction(detectionF);
}

HogSvmExSClassifier::HogSvmExSClassifier(QString modelFilename)
    : HogSvmExSClassifier()
{
    loadFromFile(modelFilename);
}

HogSvmExSClassifier::~HogSvmExSClassifier()
{
    delete svm;
    delete descriptor;
    delete &exSearch;
}

bool HogSvmExSClassifier::predict(Mat img)
{
    exSearch.setWidth(img.cols);
    return !(exSearch(img).isEmpty());
}

bool HogSvmExSClassifier::loadFromFile(QString filename)
{
    qDebug() << "Ladowanie modelu z ";
    qDebug() << filename;
    //filename="/home/marcin/build-ClassificationLab-Desktop-Debug/svm_model.xml";
    svm->load(filename.toStdString().c_str());
    qDebug() << "ok";
}
