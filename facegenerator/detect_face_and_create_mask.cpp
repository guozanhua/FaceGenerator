//
//  main.cpp
//  OpenCVGL1.1
//
//  Created by Iwami kazuya on 2014/11/04.
//  Copyright (c) 2014年 kazuya. All rights reserved.
//


//ここ参照
//http://nantekottai.com/2014/04/16/opencv-xcode5-homebrew/

//#include <iostream>
#include <core.hpp>
#include <highgui.hpp>
#include <objdetect.hpp>
#include <imgproc.hpp>
#include <stdio.h>

int size_of_mosaic = 0;

int main (int argc, char **argv)
{
    cv::Mat input,mask_img,not_masked;
    //loading haar classifier
    std::string cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadefile見つからん！\n");
        return -1;
    }
    
    //loading resource file(for face)
    input=cv::imread("/Users/naoto/git/opencv_gl/opencv/minematsu1.png",1);
    if(input.empty()){
        printf("ERROR: image not found!\n");
        return 0;
    }
    
    double scale = 4.0;
    cv::Mat gray, smallImg(cv::saturate_cast<int>(input.rows/scale),cv::saturate_cast<int>(input.cols/scale),CV_8UC1);
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    cv::resize(gray, smallImg, smallImg.size(),0,0,cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);//ヒストグラムビンの合計値が 255 になるようヒストグラムを正規化
    
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(smallImg, faces,1.1,2,CV_HAAR_SCALE_IMAGE,cv::Size(20,20));
    
    int i;
    printf("deteced faces:%d\n",(int)faces.size());
    for (i=0; i<faces.size(); i++) {
        cv::Point center,p1,p2;
        int radius;
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height*0.5)*scale);
        radius = cv::saturate_cast<int>((faces[i].width + faces[i].height)*0.25*scale);
        p1.x=center.x-radius;p1.y=center.y-radius;
        p2.x=center.x+radius;p2.y=center.y+radius;
        cv::Rect roi_rect(center.x-radius,center.y-radius,radius*2,radius*2);//左上のx座標,y座標,width,depthというふうに格納していく
        mask_img.create(input.size(), CV_8UC1);
        mask_img=cv::Scalar(0,0,0);
        not_masked=mask_img(roi_rect);
        not_masked=cv::Scalar(255,255,255);
    }
    
    cv::namedWindow("result",1);
    cv::namedWindow("masked",1);
    cv::imshow("result", input);
    cv::imshow("masked", mask_img);
    cv::waitKey(0);
    return 0;
    
}

