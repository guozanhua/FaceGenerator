#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#define WIDTH 1280
#define HEIGHT 640

int main(int argc, char *argv[])
{
    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
    if(!cap.isOpened())
    {
        printf("カメラが検出できませんでした");
        return -1;
    }
    cv::Mat input_img;
    cv::Mat hsv_skin_img= cv::Mat(cv::Size(WIDTH,HEIGHT),CV_8UC3);
    cv::Mat smooth_img;
    cv::Mat hsv_img;
    
    cv::namedWindow("input_img", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("hsv_skin_img", CV_WINDOW_AUTOSIZE);
    
    while(1)
    {
        hsv_skin_img = cv::Scalar(0,0,0);
        cap >> input_img;
        cv::medianBlur(input_img,smooth_img,7);	//ノイズがあるので平滑化
        cv::cvtColor(smooth_img,hsv_img,CV_BGR2HSV);	//HSVに変換
        for(int y=0; y<HEIGHT;y++)
        {
            for(int x=0; x<WIDTH; x++)
            {
                int a = hsv_img.step*y+(x*3);
                if(hsv_img.data[a] >=0 && hsv_img.data[a] <=15 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 50 ) //HSVでの検出
                {
                    hsv_skin_img.data[a] = 255; //肌色部分を青に
                }
            }
        }
        cv::imshow("input_img",input_img);
        cv::imshow("hsv_skin_img",hsv_skin_img);
        if(cv::waitKey(30) >=0)
        {
            break;
        }
    }
}