#-------------------------------------------------
#
# Project created by QtCreator 2016-01-13T15:12:32
#
#-------------------------------------------------
#-------------------------------------------------

QT       += core
QT       += network
QT       -= gui
QT       += sql

CONFIG   += console
CONFIG   -= app_bundle
CONFIG   += c++11

TEMPLATE = app

#INCLUDEPATH += /home/marcin/caffe/distribute/include
#INCLUDEPATH += /user/local/include
#INCLUDEPATH += /home/marcin/caffe/include

#LIBS += 'pkg-config opencv --libs'
#LIBS += -LC:/home/marcin/opencv-2.4.9
LIBS += -L/usr/local/lib
LIBS += -L/usr/lib/x86_64-linux-gnu
#LIBS += -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249 -lopencv_features2d249 -lopencv_photo249 -lopencv_ml249 -lopencv_objdetect249
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_photo -lopencv_ml -lopencv_objdetect

LIBS += -L/home/marcin/caffe/build/lib
LIBS += -lcaffe -lprotobuf -lpthread -lglog
TARGET = ClassificationLab
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    training.cpp \
    predicting.cpp \
    classifier.cpp \
    hogclassifier.cpp \
    defaulthogclassifier.cpp \
    autotest.cpp \
    treeclassifier.cpp \
    treehogclassifier.cpp

HEADERS += \
    utils.hpp \
    essentials.hpp \
    classifier.h \
    hogclassifier.hpp \
    defaulthogclassifier.hpp \
    common.hpp \
    treeclassifier.hpp \
    treehogclassifier.hpp
