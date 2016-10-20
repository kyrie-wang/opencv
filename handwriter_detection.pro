#-------------------------------------------------
#
# Project created by QtCreator 2016-10-15T21:41:41
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = handwriter_detection
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

HEADERS += \
    mnist.h

INCLUDEPATH += \
    /usr/local/include/


LIBS += \
    -I/usr/local/include \
    -ltesseract \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/lib \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml  \


