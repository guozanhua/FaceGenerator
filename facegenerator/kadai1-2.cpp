//
//  kadai1-2.cpp
//  test.cpp
//
//  Created by 井上直人 on 2014/11/06.
//  Copyright (c) 2014年 井上直人. All rights reserved.
//

//for Mac
// TODO スクショ
#include <core.hpp>
#include <photo.hpp>
#include <highgui.hpp>
#include <stdio.h>

cv::Mat original_image, selected_image,reverse_image;
bool selected=false;

cv::Point prev_pt;
cv::Point prev_end_pt;

void on_mouse(int event, int x , int y , int flags, void *){
    if(selected_image.empty()){
        return;
    }
    if(event == CV_EVENT_LBUTTONDOWN && !selected){
        prev_pt=cv::Point(x,y);
        // set the start point
    }else if(event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) && !selected){
        //消して、再描画
        cv::Point pt(x, y);
        if(prev_pt.x < 0){
            prev_pt = pt;
        }
        original_image.copyTo(selected_image);//いっぺん元の絵を再読み込み
        cv::rectangle(selected_image, prev_pt, pt, cv::Scalar::all(255),1,8,0);
        //cv::Mat img_hdr =selected_image;
        cv::imshow("image",selected_image);
    }else if (event == CV_EVENT_LBUTTONUP && !selected){
        prev_end_pt=cv::Point(x,y);
        selected=true;
    }else if (event==CV_EVENT_LBUTTONDOWN && selected){
        original_image.copyTo(selected_image);
        cv::imshow("image", selected_image);
    }else if (event==CV_EVENT_LBUTTONUP && selected){
        selected=false;
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
           "\tESC or q - quit the program\n"
           "\tr - reverse mode\n"
           "\tz - undo reverse");
    
    // 2. prepare window
    cv::namedWindow("image",1);
    
    // 3. prepare Mat objects for processing-mask and processed-image
    selected_image=original_image.clone();                           // Matオブジェクトのクローンを生成する
    // 3. prepare Mat objects for processing-mask and processed-image
    // Matオブジェクトのクローンを生成する
    reverse_image=original_image.clone();//
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
            case 'r':{//{}にしないと、二つの変数のスコープがswitch文内になってしまうため、初期化されない可能性が
                //http://blog.livedoor.jp/bribser_dev/archives/4181608.html
                //矩形の一部分を変数に代入してくる。ポインタ渡し的なやつなので、元のデータの内容も書きかわる(浅いコピー?)
                //http://book.mynavi.jp/support/pc/opencv2/c3/opencv_img.html
                //ネガポジの反転
                //cv::Mat(selected_data,selected_area) というのをマスクすれば良いのかな?
                cv::Rect selected_area(prev_pt,prev_end_pt);
                cv::Mat buf_input = selected_image(selected_area);
                cv::Mat buf_output = reverse_image(selected_area);
                cv::bitwise_not(buf_input, buf_output);
                cv::imshow("image", reverse_image);
                break;}
            case 'z'://ctrl+zだと思ってもらえれば 選択状態へ戻る
                cv::imshow("image", selected_image);
                reverse_image=original_image.clone();
                selected=true;
                break;
        }
    }
    return 0;
}

