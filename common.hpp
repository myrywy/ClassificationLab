#ifndef COMMON_HPP
#define COMMON_HPP

#include "essentials.hpp"

std::vector<float> getHogDescriptor(QString filename);
QVector<cv::Mat> createInputMats(QList<QDir> clasesDirs);
int countImgInDir(QString dirPath);

#endif // COMMON_HPP
