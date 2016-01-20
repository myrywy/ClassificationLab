#include "treehogclassifier.hpp"

TreeHogClassifier::TreeHogClassifier()
{
    trees=new CvGBTrees;
    descriptor=new HOGDescriptor;
    descriptor->winSize=cv::Size(160,96);
}

TreeHogClassifier::TreeHogClassifier(QString modelfile)
    :TreeHogClassifier()
{
    loadFromFile(modelfile);
}

TreeHogClassifier::~TreeHogClassifier()
{
    delete trees;
    delete descriptor;
}

bool TreeHogClassifier::predict(cv::Mat img)
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
    qDebug() << "Trees";
    float fRes=trees->predict(testInput);
    qDebug() << "Klasa: " << fRes;
    if(fRes > 0.1){
        return true;
    }else{
        return false;
    }
}

bool TreeHogClassifier::loadFromFile(QString filename)
{
    qDebug() << "Ladowanie modelu z ";
    qDebug() << filename;
    //filename="/home/marcin/build-ClassificationLab-Desktop-Debug/svm_model.xml";
    trees->load(filename.toStdString().c_str());
    qDebug() << "ok";
}
