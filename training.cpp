#include "utils.hpp"
#include "essentials.hpp"
#include "common.hpp"

#include <QDir>
#include <QFile>

typedef QList< QList<QString> > ClasesFiles;
typedef QList< std::vector<float> > ListOfDesciptors;

using namespace cv;

int winSizeH=96;
int winSizeW=160;

int training(){
    qDebug() << "Training: start";
    QFile options("clases.dat");
    if(!options.open(QFile::ReadOnly | QFile::Text)){
        return -1;
    }
    winSizeW=QString(options.readLine()).trimmed().toInt();
    winSizeH=QString(options.readLine()).trimmed().toInt();
    qDebug() << "Size: "<< winSizeW << " x " << winSizeH;
    bool ok=!options.atEnd();
    if(!ok){
        qDebug() << "No clases";
        return -1;
    }
    QList<QDir> clasesDirs;
    while(ok){
        QString classDir=options.readLine();
        classDir=classDir.trimmed();
        clasesDirs.append(QDir(classDir));
        qDebug() << "Class dir: " << classDir;
        ok=!options.atEnd();
    }

    qDebug() << "Training start";

    QVector<Mat> trainMats=createInputMats(clasesDirs);
    Mat trainInput=trainMats[0];
    Mat trainResponses=trainMats[1];

    //imshow("deskryptory",trainInput);
    //imwrite("deskryptory.png",trainInput);
    //imwrite("odpowiedzi.png",trainResponses);

    cv::SVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    //params.kernel_type = CvSVM::LINEAR;
    //params.kernel_type = CvSVM::POLY;
    //params.kernel_type = CvSVM::SIGMOID;
    //params.kernel_type = CvSVM::RBF;
    //params.degree=2;
    //params.gamma=1;
    //params.coef0=1;
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    //params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1, 1e-6);
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1, 0.1);

    qDebug() << "SVM in progress";
    CvSVM svm;
    //svm.train(trainInput,trainResponses,Mat(),Mat(),params);
    svm.train(trainInput,trainResponses);
    svm.save("svm_model.xml");
    qDebug() << "SVM saved";
    qDebug() << "Trees in progress";
    CvGBTrees tree(trainInput,CV_ROW_SAMPLE,trainResponses);
    tree.save("tree_model.xml");
    qDebug() << "Trees saved";

    return 0;
    /*
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
    tree.save("tree_model");
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

    return 0;*/
}

int countImgInDir(QString dirPath){
    //qDebug() << "counting " << dirPath;
    QDir dir(dirPath);
    int counter=0;
    for(QString filename : dir.entryList()){
        if(filename.contains(".jpg") || filename.contains(".JPG")){
            if(filename!="." || filename!=".."){
                counter++;
            }
        }
    }
    //qDebug() << "counting " << counter;
    return counter;
}

QVector<Mat> createInputMats(QList<QDir> clasesDirs){
    QVector<int> examplesCount(clasesDirs.size());
    int classIndex=0;
    for(QDir &classDir : clasesDirs){
        examplesCount[classIndex]=0;
        auto subDirsList=classDir.entryInfoList();
        for(QFileInfo& dir : subDirsList){
            if(!dir.isDir()){
                continue;
            }
            if(dir.fileName()=="." || dir.fileName()==".."){
                continue;
            }
            if(!dir.fileName().contains("R")){
                continue;
            }
            //qDebug() << "dir.path()=" << dir.path();
            //qDebug() << "countImgInDir(dir.path())=" << countImgInDir(dir.path());
            examplesCount[classIndex]+=countImgInDir(dir.path()+"/"+dir.fileName());
        }
        ++classIndex;
    }
    classIndex=0;
    int totalExamples=0;
    for(int &i : examplesCount){
        totalExamples+=i;
    }
    QVector< std::vector<float> > descriptorsValues(totalExamples);
    QVector< int > responses(totalExamples);
    int exampleIndex=0;
    for(QDir &classDir : clasesDirs){
        qDebug() << "Class " << classIndex << " in progress...";
        examplesCount[classIndex]=0;
        auto subDirsList=classDir.entryInfoList();
        for(QFileInfo& dir : subDirsList){
            //dla kazdego podkatalogu w katalogu klasy
            if(!dir.isDir()){
                continue;
            }
            if(dir.fileName()=="." || dir.fileName()==".."){
                continue;
            }
            if(!dir.fileName().contains("R")){
                continue;
            }
            //qDebug() << "Working in directory:";
            //qDebug() << dir.fileName();

            for(QString filename : QDir(dir.path()+"/"+dir.fileName()).entryList()){
                //dla każdego obrazka
                if(filename.contains(".jpg") || filename.contains(".JPG")){
                    if(filename!="." || filename!=".."){
                        //qDebug() << "Working on:";
                        //qDebug() << filename;
                        descriptorsValues[exampleIndex]=getHogDescriptor(dir.path()+"/"+dir.fileName()+"/"+filename);
                        responses[exampleIndex]=classIndex;
                        exampleIndex++;
                    }
                }
            }
        }
        ++classIndex;
    }
    int inCols=descriptorsValues[0].size();
    Mat trainInput(totalExamples,inCols,CV_32FC1);
    Mat trainResponses(totalExamples,1,CV_32SC1);

    classIndex=0;
    exampleIndex=0;
    for(std::vector<float> descriptor : descriptorsValues){
        //Dla każdego deskryptora czyli rzedu macierzy
        for(int j=0; j<inCols; j++){
            trainInput.at<float>(exampleIndex,j)=descriptor[j];
        }
        trainResponses.at<int>(exampleIndex,0)=responses[exampleIndex];
        exampleIndex++;
    }
    QVector<Mat> res;
    res.append(trainInput);
    res.append(trainResponses);
    return res;
}

std::vector<float> getHogDescriptor(QString filename){
    qDebug() << filename;
    Mat img = imread(filename.toStdString());
    /*imshow("ghd", img);
    waitKey(3000);*/
    HOGDescriptor descriptor;
    /*qDebug() << img.cols << " x " << img.rows;
    qDebug() << descriptor.winSize.width << " x " << descriptor.winSize.height;
    qDebug() << descriptor.blockSize.width << " x " << descriptor.blockSize.height;
    qDebug() << descriptor.blockStride.width << " x " << descriptor.blockStride.height;
    qDebug() << descriptor.cellSize.width << " x " << descriptor.cellSize.height;*/
    descriptor.winSize=Size(winSizeW,winSizeH);
    std::vector<float> tmp;
    descriptor.compute(img,tmp);
    return tmp;
}
