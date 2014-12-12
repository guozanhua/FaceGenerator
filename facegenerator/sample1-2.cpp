//
//  sample1-2.cpp
//  test.cpp
//
//  Created by 井上直人 on 2014/11/06.
//  Copyright (c) 2014年 井上直人. All rights reserved.
//
//for Mac
#include <core.hpp>
#include <photo.hpp>
#include <highgui.hpp>
#include <stdio.h>

cv::Mat inpaint_mask;
cv::Mat original_image, selected_image, inpainted;

cv::Point prev_pt;

void on_mouse(int event, int x , int y , int flags, void *){
    if(selected_image.empty()){
        return;
    }
    
    if(event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON)){
        prev_pt=cv::Point(-1,-1);                          // init the start point
    }
    else if(event == CV_EVENT_LBUTTONDOWN){
        prev_pt=cv::Point(x,y);                         // set the start point
    }
    else if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)){
        cv::Point pt(x, y);
        if(prev_pt.x < 0){
            prev_pt = pt;
        }
        
        // draw a line from the start point to the current point
        cv::line(inpaint_mask, prev_pt, pt, cv::Scalar(255),5,8,0);                           // 直線の描画を行なう
        cv::line(selected_image, prev_pt, pt, cv::Scalar::all(255),5,8,0);
        
        // set the current point to the new start point
        prev_pt = pt;
        
        //cv::Mat img_hdr =selected_image;
        cv::imshow("image",selected_image);
    }
}

int main(int argc, char *argv[]){
    
    // 1. read image file
    char *filename = (argc >= 2) ? argv[1] : (char*)"/Users/naoto/git/opencv_gl/opencv/fruits.jpg";
    original_image = cv::imread(filename);//多分ここら辺はすべてポインタ渡し。だからコピーする時は注意
    if(original_image.empty()){
        printf("ERROR: image not found!\n");
        return 0;
    }
    
    //print hot keys
    printf( "Hot keys: \n"
           "\tESC - quit the program\n"
           "\ti or ENTER - run inpainting algorithm\n"
           "\t\t(before running it, paint something on the image)\n");
    
    // 2. prepare window
    cv::namedWindow("image",1);
    
    // 3. prepare Mat objects for processing-mask and processed-image
    selected_image=original_image.clone();                           // Matオブジェクトのクローンを生成する
    inpainted=original_image.clone();//
    inpaint_mask.create(original_image.size(),CV_8UC1);                   //
    
    inpaint_mask=cv::Scalar(0);                           // 全ての画素の値を0に初期化
    inpainted=cv::Scalar(0);
    
    // 4. show image to window for generating mask
    cv::imshow("image", selected_image);
    
    // 5. set callback function for mouse operations
    cv::setMouseCallback("image", on_mouse,0);// マウス操作のコールバック関数を登録 3つ目の引数は
    //http://opencv.jp/opencv-2.2/py/highgui_user_interface.html どれがどれかは教科書見て
    
    bool loop_flag = true;
    while(loop_flag){
        
        // 6. wait for key input
        int c = cv::waitKey(0);
        
        // 7. process according to input
        switch(c){
            case 27://ESC
            case 'q':
                loop_flag = false;
                break;
                
            case 'r':
                inpaint_mask = cv::Scalar(0);
                original_image.copyTo(selected_image);
                cv::imshow("image", selected_image);
                break;
                
            case 'i':
            case 10://ENTER
                cv::namedWindow("inpainted image", 1);
                cv::inpaint(selected_image,inpaint_mask,inpainted,3.0,cv::INPAINT_TELEA);                           // インペイント処理を行なう
                cv::imshow("inpainted image", inpainted);
                break;
        }
    }
    return 0;
}

