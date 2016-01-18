#include "utils.hpp"
#include "essentials.hpp"

typedef QList< QList<QString> > ClasesFiles;
typedef QList< std::vector<float> > ListOfDesciptors;

using namespace cv;

int training(){
    int classIndex=0;
    ClasesFiles files;
    forever{
        std::string decision;
        std::cout << "Podaj folder klasy\n";
        std::cin >> decision;
        if(decision=="fin"){
            break;
        }
        files.append(QList<QString>());
        QString folder(decision.c_str());
        std::cout << "Podaj ilosc zdjec\n";
        std::cin >> decision;
        int n=QString(decision.c_str()).toInt();
        for(int i=1; i<=n; i++){
            files[classIndex].append(folder+"/"+QString::number(i)+".jpg");
            //std::cout << files[classIndex].last().toStdString();
        }
        classIndex++;
    }
    QList<ListOfDesciptors> clasesDescriptors;
    for(int i=0; i<classIndex; i++){
        clasesDescriptors.append(ListOfDesciptors());
        for(QString s : files[i]){
            Mat img = imread(s.toStdString());
            HOGDescriptor descriptor;
            descriptor.winSize=Size(40,40);
            clasesDescriptors[i].append(std::vector<float>());
            descriptor.compute(img,clasesDescriptors[i].last());
            //qDebug() << QVector<float>::fromStdVector(clasesDescriptors[i].last());
            //imshow(s.toStdString(),img);
        }
    }
    int inRows=0;
    for(ListOfDesciptors list : clasesDescriptors){
        inRows+=list.size();
    }
    int inCols=clasesDescriptors[0][0].size();
    Mat trainInput(inRows,inCols,CV_32FC1);
    Mat trainResponses(inRows,1,CV_32SC1);
    int row=0;
    int classLabel=0;
    for(ListOfDesciptors& classDesc: clasesDescriptors){
        for(std::vector<float> descriptor : classDesc){
            //Dla każdego deskryptora czyli rzedu macierzy
            for(int j=0; j<inCols; j++){
                trainInput.at<float>(row,j)=descriptor[j];
            }
            trainResponses.at<int>(row,0)=classLabel;
            row++;
        }
        classDesc.clear();
        classLabel++;
    }
    imshow("input",trainInput);
    CvSVM svm(trainInput,trainResponses);
    svm.save("svm_model");
    QList<QString> tests;
    tests.append("test/kola/1.jpg");
    tests.append("test/kwadraty/2.jpg");
    for(QString testPath : tests){
        Mat image=imread(testPath.toStdString());
        HOGDescriptor descriptor;
        descriptor.winSize=Size(40,40);
        std::vector<float> descriptorValues;
        descriptor.compute(image,descriptorValues);
        Mat testInput(1,inCols,CV_32FC1);
        for(int i=0;i<inCols;i++){
            testInput.at<float>(0,i)=descriptorValues[i];
        }
        float label=svm.predict(testInput);
        imshow(QString::number(label).toStdString(),image);
        qDebug() << testPath << " label: " << QString::number(label);
    }


    CvGBTrees tree(trainInput,CV_ROW_SAMPLE,trainResponses);
    tree.save("svm_model");
    //QList<QString> tests;
    //tests.append("test/kola/2.jpg");
    //tests.append("test/kwadraty/2.jpg");
    for(QString testPath : tests){
        Mat image=imread(testPath.toStdString());
        HOGDescriptor descriptor;
        descriptor.winSize=Size(40,40);
        std::vector<float> descriptorValues;
        descriptor.compute(image,descriptorValues);
        Mat testInput(1,inCols,CV_32FC1);
        for(int i=0;i<inCols;i++){
            testInput.at<float>(0,i)=descriptorValues[i];
        }
        float label=tree.predict(testInput);
        imshow(QString::number(label).toStdString(),image);
        qDebug() << testPath << " label: " << QString::number(label);
    }

    return 0;
}
