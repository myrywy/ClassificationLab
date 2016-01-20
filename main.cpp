#include <QCoreApplication>

#include "utils.hpp"
#include <iostream>
#include <QFile>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QFile options("mode.dat");
    if(!options.open(QFile::ReadOnly | QFile::Text)){
        return -1;
    }
    QString decision=options.readLine();
    decision=decision.trimmed();
    options.close();
    if(decision=="train"){
        training();
    }else if(decision=="test"){
        predicting();
    }else if(decision=="autotest"){
        test();
    }else{
        return 0;
    }

    return a.exec();
}
