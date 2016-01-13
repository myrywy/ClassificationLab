#include <QCoreApplication>

#include "utils.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    std::string decision;
    std::cout << "train/predict\n";
    std::cin >> decision;
    if(decision=="train"){
        training();
    }else if(decision=="predict"){
        predicting();
    }else{
        return 0;
    }

    return a.exec();
}
