//
//  main.cpp
//  test.cpp
//
//  Created by 井上直人 on 2014/11/06.
//  Copyright (c) 2014年 井上直人. All rights reserved.
//
// 再生時は edit scheme→arfument→＋→Users/naoto/Movies/sample_video_input.avi　と入力することで引数を設定
#include <core.hpp>
#include <highgui.hpp>

#include <stdio.h>

#define IN_VIDEO_FILE "/Users/naoto/Movies/sample_video_input.avi"
#define OUT_VIDEO_FILE "/Users/naoto/Movies/sample_video_output.avi"

int main(int argc, char *argv[]){
    
    // 1. prepare VideoCapture Object
    cv::VideoCapture cap;                           // キャプチャ用のオブジェクトを用意する
    std::string input_index;
    if(argc >= 2){ // capture from video file
        input_index = argv[1];
        cap.open(input_index);                           // ファイルからのキャプチャを開始する
    }else{ // capture from camera
        cap.open(0);                           // カメラからのキャプチャを開始する
    }
    
    // 2. prepare VideoWriter Object
    cv::Mat frame, copy_frame;
    int rec_mode= 0;
    
    cv::namedWindow("video", 1);
    cv::VideoWriter output_video;                           // 録画用のオブジェクトを用意する
    output_video.open(OUT_VIDEO_FILE, CV_FOURCC('W', 'R', 'L', 'E'), 10, cv::Size(1280,720));                           // 動画の保存のための初期化を行なう
    /* using "WRLE" as the video codec */
    
    if(!cap.isOpened() || !output_video.isOpened()){
        printf("no input video\n");
        return 0;
    }
    else
    {
        bool loop_flag = true;
        while(loop_flag){
            
            // 3. capture frame from VideoCapture
            cap >> frame;                           // キャプチャを行なう
            if(frame.empty()){
                break;
            }
            
            // 4. save frame
            if(rec_mode){
                output_video << frame;                 // 録画モードであればキャプチャしたフレームを保存する
                frame.copyTo(copy_frame);              //
                cv::Size s=frame.size();               //
                cv::rectangle(copy_frame, cv::Point(0,0), cv::Point(s.width-1,s.height-1), cv::Scalar(0,0,255),4,8,0);
                
                cv::imshow("video",copy_frame);
            }
            else{
                cv::imshow("video",frame);
            }
            
            // 5. process according to input key
            int k = cvWaitKey(33);
            switch(k){
                case 'q':
                case 'Q':
                    loop_flag = false;
                    break;
                case 'r':
                    if(rec_mode ==0){
                        rec_mode = 1;
                    }else{
                        rec_mode = 0;
                    }
                    break;
            }
        }
    }
    return 0;
}

