///Users/naoto/Movies/sample_video_input.avi を引数にとって実行
#include <core.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int INIT_TIME = 50;
    int width=1280;
    int height=720;
    double B_PARAM = 1.0 / 50.0;
    double T_PARAM = 1.0 / 200.0;
    double Zeta = 10.0;
    
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat avg_img, sgm_img;
    cv::Mat lower_img, upper_img, tmp_img;
    cv::Mat dst_img, msk_img;
    
    // 1. initialize VideoCapture
    if(argc >= 2){
        cap.open(argv[1]);
    }else{
        cap.open(0);
    }
    if(!cap.isOpened()){
        printf("Cannot open the video.\n");
        exit(0);
    }
    
    // 2. prepare window for showing images
    cv::namedWindow("Input", 1);
    cv::namedWindow("FG", 1);
    cv::namedWindow("mask", 1);
    
    // 3. calculate initial value of background
    cap >> frame;
    
    cv::Size s = frame.size();
    
    avg_img.create(s, CV_32FC3);
    sgm_img.create(s, CV_32FC3);
    lower_img.create(s, CV_32FC3);
    upper_img.create(s, CV_32FC3);
    tmp_img.create(s, CV_32FC3);
    
    dst_img.create(s, CV_8UC3);
    msk_img.create(s, CV_8UC1);
    
    printf("Background statistics initialization start\n");
    
    avg_img = cv::Scalar(0,0,0);
    
    for( int i = 0; i < INIT_TIME; i++){
        cap >> frame;
        cv::Mat tmp;
        frame.convertTo(tmp, avg_img.type());                          //  入力配列に対してスケーリングを行う
        //tmp.create(avg_img.size,CV_32FC3);
        //tmp.rows=0つまり、もう読んでる画像がないってこと　だからこれを外から書き換えたりするとかではなく、強制終了
        if (tmp.rows==0) {
            break;
        }
        cv::accumulate(tmp, avg_img);                           // 画像全体を累算器に加える
    }
    
    avg_img.convertTo(avg_img, -1,1.0 / INIT_TIME);
    avg_img = cv::Scalar(0,0,0);
    
    for( int i = 0; i < INIT_TIME; i++){
        cap >> frame;
        frame.convertTo(tmp_img, avg_img.type());// 背景の輝度振幅の初期値を計算する
        if (tmp_img.rows==0) {
            break;
        }
        cv::subtract(tmp_img, avg_img, tmp_img);
        cv::pow(tmp_img, 2.0, tmp_img);
        tmp_img.convertTo(tmp_img, -1,2.0);
        cv::sqrt(tmp_img, tmp_img);
        cv::accumulate(tmp_img, sgm_img);
    }
    
    sgm_img.convertTo(sgm_img, -1,1.0/INIT_TIME);// 入力配列に対してスケーリングを行う
    
    printf("Background statistics initialization finish\n");
    
    
    bool loop_flag = true;
    while(loop_flag){
        cap >> frame;
        frame.convertTo(tmp_img, tmp_img.type());                           // 入力配列に対して変換を行う
        
        // 4. check whether pixels are background or not
        cv::subtract(avg_img, sgm_img, lower_img);          // 背景となりうる画素の輝度値の範囲をチェックする
        cv::subtract(lower_img, Zeta, lower_img);
        cv::add(avg_img, sgm_img, upper_img);
        cv::add(upper_img, Zeta, upper_img);
        if (tmp_img.rows==0) {
            break;
        }
        cv::inRange(tmp_img, lower_img, upper_img, msk_img);
        
        // 5. recalculate
        cv::subtract(tmp_img, avg_img, tmp_img);// 背景と判断された領域の背景の輝度平均と輝度振幅を更新する
        cv::pow(tmp_img, 2.0, tmp_img);
        tmp_img.convertTo(tmp_img, -1,2.0);
        cv::pow(tmp_img, 0.5, tmp_img);
        
        // 6. renew avg_img and sgm_img
        cv::accumulateWeighted(frame, avg_img, B_PARAM,msk_img);// 関数cv::accumulateWeightedにより輝度平均と輝度振幅を更新する
        cv::accumulateWeighted(tmp_img, avg_img, B_PARAM,msk_img);
        
        cv::bitwise_not(msk_img, msk_img);// 物体領域と判断された領域では輝度振幅のみを更新する
        cv::accumulateWeighted(tmp_img, avg_img, T_PARAM,msk_img);
        
        dst_img = cv::Scalar(0);
        frame.copyTo(dst_img, msk_img);
        
        cv::imshow("Input", frame);
        cv::imshow("FG", dst_img);
        cv::imshow("mask", msk_img);
        
        char key =cv::waitKey(10);
        if(key == 27){
            loop_flag = false;
        }
    }
    return 0;
}