//index_finger_detector.cpp
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <deque>

#define MAX_LIMIT 3
#define MIN_LIMIT 1.5
#define FACE_NEAR 4
#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 320
#define SIMILAR_LENGTH 16 //width=640,height=320の時の値なので割と難しいかもしれない
#define POINT_LENGTH 10

//double current_time(){
//    struct timeval tp;
//    gettimeofday(&tp,NULL);
//    return tp.tv_sec+tp.tv_usec*1.0E-6;
//}

int get_area_code(cv::Point p){//(99,30)だったら, 4+32*1=36
    assert(p.x>=0 && p.y>=0 && p.x<=VIDEO_WIDTH && p.y<=VIDEO_HEIGHT);
    int buf=(p.x/SIMILAR_LENGTH)+(VIDEO_WIDTH/SIMILAR_LENGTH)*(p.y/SIMILAR_LENGTH);
    assert(buf>=0 && buf<(VIDEO_WIDTH/SIMILAR_LENGTH)+(VIDEO_WIDTH/SIMILAR_LENGTH)*(VIDEO_HEIGHT/SIMILAR_LENGTH));
    return buf;
}
cv::Point get_area_pos(int area_code){
    cv::Point p;
    p.x=(area_code%(VIDEO_WIDTH/SIMILAR_LENGTH))*SIMILAR_LENGTH+(int)(0.5*SIMILAR_LENGTH);
    p.y=(area_code/(VIDEO_WIDTH/SIMILAR_LENGTH))*SIMILAR_LENGTH+(int)(0.5*SIMILAR_LENGTH);
    return p;
}

int specify_area(std::deque<cv::Point> points){//基本は多数決
    int max_times=1;
    int max_area=0;
    int size=(int)points.size();
    std::deque<int> areas;
    for (int i=0; i<size;i++){
        areas.push_back(get_area_code(points[i]));
    }
    for (int i=0;i<size;i++){
        int buf=areas[i];
        //printf("%d ",buf);
        int count=1;
        for (int j=i+1;j<size;j++){
            if (buf==areas[j]){count++;}
        }
        if (count>max_times){
            max_times=count;
            max_area=areas[i];
        }
    }
    return max_area;
}

cv::Point calc_grav(std::vector<cv::Point> points){
    cv::Point grav_point;
    //削除対象を見つけるループ
    double sum_x=0;
    double sum_y=0;
    int N=0;
    for(std::vector<cv::Point>::iterator itiElement=points.begin();itiElement!=points.end();itiElement++){
        cv::Point p= *itiElement;
        sum_x+=p.x;
        sum_y+=p.y;
        N++;
    }
    grav_point.x=sum_x/(double)N;
    grav_point.y=sum_y/(double)N;
    return grav_point;
}
double calc_distance(cv::Point p1,cv::Point p2){
    double distance=sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
    return distance;
}
double calc_inner_product(cv::Point p1,cv::Point p2,cv::Point p3){//(p1-p3)・(p2-p3)を計算
    double sum=(p1.x-p3.x)*(p2.x-p3.x)+(p1.y-p3.y)*(p2.y-p3.y);
    return sum;
}
bool is_near_face(cv::Point face_center,int face_radius,cv::Point finger_point){
    if (face_center.x-FACE_NEAR*face_radius<finger_point.x && face_center.x+FACE_NEAR*face_radius>finger_point.x
        && face_center.y-FACE_NEAR*face_radius<finger_point.y && face_center.y+FACE_NEAR*face_radius>finger_point.y){
        return true;
    }else{
        return false;
    }
}

std::vector<cv::Point> get_finger_point(cv::Point grav_center,cv::Point grav_top,std::vector<cv::Point> tops){
    double min_dist=10000;
    for(std::vector<cv::Point>::iterator itiElement=tops.begin();itiElement!=tops.end();itiElement++){
        double buf_dist=calc_distance(grav_center,*itiElement);
        if (buf_dist<min_dist) {
            min_dist=buf_dist;
        }
    }
    std::vector<cv::Point> bufs;
    cv::Point max_point;
    double max_dist=0;
    for(std::vector<cv::Point>::iterator itiElement=tops.begin();itiElement!=tops.end();itiElement++){
        double buf_dist=calc_distance(grav_center,*itiElement);
        if(buf_dist>max_dist && buf_dist>=MIN_LIMIT*min_dist && buf_dist<=MAX_LIMIT*min_dist){
            if (calc_inner_product(*itiElement, grav_top,grav_center)>0){//grav_center→grav_topの向きが手の先の方へ向かう向きだから！　内積>0で対応
                max_dist=buf_dist;
                max_point=*itiElement;
            }
        }
    }
    
    bufs.push_back(max_point);
    return bufs;
}

std::vector<cv::Rect> get_faces(cv::Mat input_img,double scale,cv::CascadeClassifier cascade){
    cv::Mat gray, smallImg(cv::saturate_cast<int>(input_img.rows/scale),cv::saturate_cast<int>(input_img.cols/scale),CV_8UC1);
    cv::cvtColor(input_img, gray, CV_BGR2GRAY);
    cv::resize(gray, smallImg, smallImg.size(),0,0,cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);//ヒストグラムビンの合計値が 255 になるようヒストグラムを正規化
    
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(smallImg,faces,1.1,2,CV_HAAR_SCALE_IMAGE,cv::Size(20,20));
    return faces;
}
int main(int argc, char *argv[])
{
    //setting up video
    cv::VideoCapture cap;
    cap.open(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);
    if(!cap.isOpened()){printf("カメラが検出できませんでした");return -1;}
    //setting up classifier
    double scale=4.0;
    std::string cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadefile見つからん！\n");
        return -1;
    }
    
    //processing
    cv::Mat hsv_skin_img= cv::Mat(cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT),CV_8UC3);
    cv::Mat hsv_img,gray_img,bin_img,input_img,smooth_img,dst_img,dst_img_norm;
    
    cv::namedWindow("input_img", CV_WINDOW_AUTOSIZE);
    //cv::Point tmp;tmp.x=0;tmp.y=0;
    std::deque<cv::Point> finger_points;
    //finger_points.push_back(tmp);//initializing(prevent initial error)
    std::deque<int> area_code;
    //area_code.push_back(0);
    while(1)
    {
        // 手の検出のための前処理
        hsv_skin_img = cv::Scalar(0,0,0);
        cap >> input_img;
        cv::medianBlur(input_img,smooth_img,7);	//eliminate noises
        cv::cvtColor(smooth_img,hsv_img,CV_BGR2HSV);// convert(RGB→HSV)
        for(int y=0; y<VIDEO_HEIGHT;y++){
            for(int x=0; x<VIDEO_WIDTH; x++){
                int a = hsv_img.step*y+(x*3);
                //http://momiage.net/5-meisai.shtml
                //V:明度も調整。255=白っぽいほう、0＝暗っぽい方
                if(hsv_img.data[a] >=0 && hsv_img.data[a] <=20 &&hsv_img.data[a+1] >=50 && hsv_img.data[a+2] >= 70
                   &&hsv_img.data[a+2]<=200){hsv_skin_img.data[a] = 255;}
            }
        }
        cv::Size s =hsv_skin_img.size();
        cv::cvtColor(hsv_skin_img, gray_img, CV_BGR2GRAY);
        
        //処理結果の距離画像出力用の画像領域と表示ウィンドウを確保
        dst_img.create(s,CV_32FC1);dst_img_norm.create(s,CV_8UC1);
        // (3)距離画像を計算し，表示用に結果を0-255に正規化する
        cv::distanceTransform(gray_img, dst_img,CV_DIST_L2,5);
        cv::normalize (dst_img, dst_img_norm, 0.0, 255.0, CV_MINMAX, NULL);
        std::vector<std::vector<cv::Point> > contours;
        
        // 画像の二値化
        //cv::threshold(dst_img_norm, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
        // 輪郭の検出
        cv::findContours(dst_img_norm, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        int contourid=0;
        double maxarea=0;
        if (contours.size()==0) {continue;}//if no contour
        for(int i = 0; i < contours.size(); ++i) {
            size_t count = contours[i].size();
            if(count < 300 || count > 10000) continue; // （小さすぎる|大きすぎる）輪郭を除外
            double area=fabs(cv::contourArea(contours[i]));
            if (area>maxarea){
                maxarea=area;
                contourid=i;
            }
        }
        cv::Point grav_of_hands=calc_grav(contours[contourid]);
        cv::circle(input_img, grav_of_hands,10,cv::Scalar(0,255,0),-1,8,0);
        cv::drawContours(input_img,contours,contourid,cv::Scalar(255,0,0));//第3引数は本来はリストのうちcontourに該当するものを返すのだが、
        std::vector<cv::Point> tops,candidacies;
        cv::convexHull(contours[contourid],tops);
        //http://opencv.jp/opencv-2.1/c/structural_analysis_and_shape_descriptors.html#cvconvexitydefect うまく実装できねえ..
        //std::vector<std::vector<cv::Point>> tops1;
        //std::vector<cv::Point> tops1;
        //cv::convexHull(contours[contourid],tops1,CV_CLOCKWISE,0);
        //std::vector<CvConvexityDefect> concavities;
        //cv::convexityDefects(contours[contourid], tops1,concavities);
        //for(std::vector<CvConvexityDefect>::iterator itiElement=concavities.begin();itiElement!=concavities.end();itiElement++){
        //    CvConvexityDefect c= *itiElement;
        //    //中心からの距離で切る?
        //    cv::circle(input_img,*c.depth_point,10,cv::Scalar(0,0,255),-1,8,0);
        //}
        cv::Point grav_of_tops=calc_grav(tops);
        cv::circle(input_img, grav_of_tops,10,cv::Scalar(0,0,255),-1,8,0);
        candidacies=get_finger_point(grav_of_hands,grav_of_tops,tops);
        for(std::vector<cv::Point>::iterator itiElement=candidacies.begin();itiElement!=candidacies.end();itiElement++){
            cv::Point p= *itiElement;
            cv::circle(input_img,p,10,cv::Scalar(0,255,0),-1,8,0);
        }
        if(cv::waitKey(100)>=0){break;}
        //過去の場所の更新
        if ((int)finger_points.size()>=POINT_LENGTH){
            finger_points.pop_front();
        }
        finger_points.push_back(candidacies[0]);
        if ((int)area_code.size()>=POINT_LENGTH){
            area_code.pop_front();
        }
        area_code.push_back(specify_area(finger_points));
        if (area_code[0]==area_code[POINT_LENGTH-1] && area_code[0]!=0){//顔の検出
            //printf("touch detected at:%d\n",area_code[0]);
            cv::Point detected_point=get_area_pos(area_code[0]);
            cv::circle(input_img,detected_point,20,cv::Scalar(255,0,0),-1,8,0);
            std::vector<cv::Rect> faces=get_faces(input_img,scale,cascade);
            bool is_near[faces.size()];
            double d[faces.size()];
            printf("%d ",(int)faces.size());
            for (int i=0; i<faces.size(); i++) {
                cv::Point center;int radius;
                //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
                center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width*0.5)*scale);//scaleはここで戻していることに注意！
                center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height*0.5)*scale);
                radius = cv::saturate_cast<int>((faces[i].width + faces[i].height)*0.25*scale);
                cv::Rect roi_rect(center.x-radius,center.y-radius,radius*2,radius*2);//左上のx座標,y座標,width,depthというふうに格納していく
                //cv::rectangle(input_img, roi_rect, cv::Scalar(0,255,0));
                is_near[i]=is_near_face(center,radius,detected_point);
                d[i]=calc_distance(center,detected_point);
            }
            int min_d=10000;
            int min_index=-1;
            for (int i=0; i<faces.size();i++) {
                if (is_near[i]==true) {
                    if (min_d>d[i]) {min_d=d[i];min_index=i;
                    }
                }
            }
            cv::Rect best_face;
            if (min_index!=-1){//条件を満たす中で最も距離の近い顔を検出できた
                best_face=faces[min_index];
                printf("detect face\n");
                cv::Point center,p1,p2;int radius;
                //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
                center.x = cv::saturate_cast<int>((faces[min_index].x + faces[min_index].width*0.5)*scale);//scaleはここで戻していることに注意！
                center.y = cv::saturate_cast<int>((faces[min_index].y + faces[min_index].height*0.5)*scale);
                radius = cv::saturate_cast<int>((faces[min_index].width + faces[min_index].height)*0.25*scale);
                cv::Rect roi_rect(center.x-radius,center.y-radius,radius*2,radius*2);//左上のx座標,y座標,width,depthというふうに格納していく
                cv::rectangle(input_img, roi_rect, cv::Scalar(255,0,0));
            }
            
        }
        cv::imshow("input_img",input_img);
    }
}