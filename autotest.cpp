#include "utils.hpp"
#include "essentials.hpp"
#include "hogclassifier.hpp"
#include "defaulthogclassifier.hpp"
#include "treehogclassifier.hpp"
#include <QDebug>
#include <QTime>

void testClass(QString classDir, Classifier* classifier, bool desiredValue, int* totalCount, int* missclassified);

int test(){
    qDebug() << "Autotest";
    QFile options("testdata.dat");
    if(!options.open(QFile::ReadOnly | QFile::Text)){
        return -1;
    }
    QString metoda=QString(options.readLine()).trimmed();
    QString modelPath=QString(options.readLine()).trimmed();
    QString dataPath=(options.readLine()).trimmed();
    Classifier* classifier;
    if(metoda=="hogSvm"){
        classifier=new HogClassifier(modelPath);
    }else if(metoda=="trees"){
        classifier=new TreeHogClassifier(modelPath);
    }else if(metoda=="defaultHog"){
        classifier=new DefaultHogClassifier;
    }
    int falsePositive{0};
    int truePositive{0};
    int falseNegative{0};
    int trueNegative{0};
    int total{0};
    QDir mainDir(dataPath);
    QFileInfoList subdirInfo=mainDir.entryInfoList();
    for(QFileInfo &fileinfo : subdirInfo){
        if(!fileinfo.isDir() || fileinfo.fileName()=="." || fileinfo.fileName()==".."){
            continue;
        }
        bool ok;
        qDebug() << fileinfo.fileName();
        int classLabel=fileinfo.fileName().toInt(&ok);
        if(ok){
            int totalCount;
            int missclassified;
            bool desiredResponse=(classLabel>0?true:false);
            qDebug() << fileinfo.filePath();
            testClass(fileinfo.filePath(),
                      classifier,
                      desiredResponse,
                      &totalCount,
                      &missclassified);
            total+=totalCount;
            if(desiredResponse){
                falseNegative+=missclassified;
                truePositive+=totalCount-missclassified;
            }else{
                falsePositive+=missclassified;
                trueNegative+=totalCount-missclassified;
            }
        }
    }
    double falsePositiveRate=double(falsePositive)/double(falsePositive+trueNegative);
    double truePositiveRate=double(truePositive)/double(truePositive+falseNegative);
    qDebug() << "calkowita liczba przykladow " << total;
    qDebug() << "false positive " << falsePositive;
    qDebug() << "true positive " << truePositive;
    qDebug() << "false negative " << falseNegative;
    qDebug() << "true negative " << trueNegative;
    qDebug() << "false positive rate " << falsePositiveRate;
    qDebug() << "true positive rate " << truePositiveRate;
    delete classifier;
    return 0;
}


void testClass(QString classDir, Classifier* classifier, bool desiredValue, int* totalCount, int* missclassified){
    *totalCount=0;
    *missclassified=0;
    QDir dir(classDir);
    for(QString filename : dir.entryList()){
        if(filename.contains(".jpg") || filename.contains(".JPG")){
            cv::Mat input=cv::imread((dir.path()+"/"+filename).toStdString());
            qDebug() << "plik: " << dir.path()+"/"+filename;
            //cv::imshow("test",input);
            //cv::waitKey(10000);
            qDebug() << QTime::currentTime().toString();
            qDebug() << QTime::currentTime().msec();
            if(classifier->predict(input)!=desiredValue){
                (*missclassified)++;
            }
            qDebug() << QTime::currentTime().toString();
            qDebug() << QTime::currentTime().msec();
            (*totalCount)++;
        }
    }

}
