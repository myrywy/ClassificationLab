#include "utils.hpp"
#include "essentials.hpp"
#include "hogclassifier.hpp"
#include "defaulthogclassifier.hpp"
#include <QDebug>

int predicting(){
    qDebug() << "Test mode";
    HogClassifier hog("/home/marcin/build-ClassificationLab-Desktop-Debug/svm_model.xml");
    //DefaultHogClassifier hog;
    bool fin=false;
    while(!fin){
        qDebug() << "Podaj sciezki plikow";
        std::string filename;
        std::cin >> filename;
        if(filename!="fin"){
            cv::Mat m=cv::imread(filename);
            std::cout << "\n" << hog.predict(m) << "\n";
        }
    }
    return 0;
}
