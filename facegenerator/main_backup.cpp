//main.cpp
//include二行と、build_targetをOSx.10.8以下,後はpathにOPENGLとGLUTを入れればOK
#include <iostream>
#include <deque>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cv.h>
#include <highgui.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>

//GL window size
#define WINDOW_X (640)
#define WINDOW_Y (640)
#define WINDOW_NAME "facegenerator" //画像については、600*600以上が読み込めなくなった
//GL texture size
#define TEXTURE_HEIGHT 540
#define TEXTURE_WIDTH 540
//GL input video size
#define VIDEO_HEIGHT 540
#define VIDEO_WIDTH 540

//to dedect finger(used at get_finger_point())
#define MAX_LIMIT 3
#define MIN_LIMIT 1.5
//to find face near finger
#define FACE_NEAR 4
//メッシュ状分割の一辺
#define SIMILAR_LENGTH 15 //width=640,height=320の時の値なので割と難しいかもしれない
//to keep the position of finger
#define POINT_LENGTH 5

//顔を含む画像の切り出しの際に、顔の縦横何倍の領域を切り出すか
#define MAGNIFICATION 3.0

//顔の傾きの許容範囲
#define ANGLE 15
// 顔の合成部分
#define LOOP_MAX 10000
//#define EPS 2.2204e-016
#define EPS 2.2204e-010
#define NUM_NEIGHBOR 4

//パーツの抽出を大きめに
#define PARTS_RATE 1.2

const char* inputFileNames[4] = {"/Users/naoto/git/opencv_gl/opencv/mona_lisa.png", "/Users/naoto/git/opencv_gl/opencv/kawagoe.jpg", "/Users/naoto/git/opencv_gl/opencv/horikita.jpg","/Users/naoto/git/opencv_gl/opencv/fukuyama.jpg"};

//global variable
int g_display_mode = 0;//0:始動時 1:flatな状態　2:画像選択を受けての反映待ち　3:videoをonにするための待機 4:videoのopen状態(手動キャプチャモード)　 5が画像取得して、合成か再取得かを待ってるモード　6合成開始モード 7 合成終了して、素材をまた選んでね！もしくは再スタートしてね！ってやるモード　8video onのための待機　9(自動キャプチャモード)
int selected_photo= 0;//which photo is selected (1~4,0=NULL)

//顔検出周りのglobal variable
double scale=4.0;
cv::CascadeClassifier cascade,nested_cascade_eye,nested_cascade_nose,nested_cascade_mouth;
//手検出周りのglobal variable
std::deque<cv::Point> finger_points;
std::deque<int> area_code;


GLuint g_TextureHandles[6] = {1,2,3,4,5,6};//5 for captured,6 for generated
cv::VideoCapture cap;
cv::Mat captured;
cv::Mat generated;
std::deque<cv::Mat> resources;//RGB系で登録されてる
bool isprinted=false;
bool iscaptured=false;
bool isgenerated=false;
//http://slis.tsukuba.ac.jp/~fujisawa.makoto.fu/lecture/iml/text/screen_character.html writing characters on OpenGL
std::string str1="choose destination image.";
std::string str2="press 1,2,3,4 to choose one";
std::string str3="press m to detect(manual), press a to detect(auto),or choose again";
std::string str4="detecting.. point the face with your finger";
std::string str5="face detected";
std::string str6="press g to generate new face,c(or a)to detect again";
std::string str7="face generating..";
std::string str8="face generated!";
std::string str9="press 1,2,3,4 to change destination image and generate";
std::string str10="press n to create new one";
std::string str11="auto detecting..";

// positions(-1~1,-1~1のはず)
double vertices[][2] = {
    {-1, 1},// pointA,0
    {0, 1},// pointB,1
    {1, 1},// pointC,2
    {-1, 0},// pointD,3
    {0, 0},// pointE,4
    {1, 0},// pointF,5
    {-1, -0.5},// pointG,6
    {-0.5, -0.5},// pointH,7
    {0, -0.5},// pointI,8
    {0.5, -0.5},// pointJ,9
    {1, -0.5},// pointK,10
    {-1, -1},// pointL,11
    {-0.5, -1},// pointM,12
    {0, -1},// pointN,13
    {0.5, -1},// pointO,14
    {1, -1},// pointP,15
};

////////////////////////////////////////////////////////////from expanding_face.cpp
int quasi_poisson_solver(IplImage *im_src, IplImage *im_dst, IplImage *im_mask, int channel, int *offset){
    int i, j, loop, neighbor, count_neighbors, ok;
    float error, sum_f, sum_vpq, fp;
    int naddr[NUM_NEIGHBOR][2]={{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    CvMat* im_new = cvCreateMat(im_dst->height, im_dst->width, CV_64F);
    for(i=0; i<im_dst->height; i++){
        for(j=0; j<im_dst->width; j++){
            cvmSet(im_new, i, j, double(uchar(im_dst->imageData[(i)*im_dst->widthStep + (j)*im_dst->nChannels + channel])));
        }
    }
    
    for(loop=0; loop<LOOP_MAX; loop++){
        if (loop%100==0) {
            printf("loop:%d\n",loop);
        }
        ok = 1;
        for(i=0; i<im_mask->height; i++){
            for(j=0; j<im_mask->width; j++){
                if(int((uchar)im_mask->imageData[i*im_mask->widthStep + j]) > 0){
                    sum_f=0.0;
                    sum_vpq=0.0;
                    count_neighbors=0;
                    for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++){
                        if(i+offset[0]+naddr[neighbor][0] >= 0 && j+offset[1]+naddr[neighbor][1] >= 0 && i+offset[0]+naddr[neighbor][0] < im_dst->height && j+offset[1]+naddr[neighbor][1] < im_dst->width){
                            sum_f += cvmGet(im_new, i+offset[0]+naddr[neighbor][0], j+offset[1]+naddr[neighbor][1]);
                            sum_vpq += float((uchar)im_src->imageData[(i)*im_src->widthStep + (j)*im_src->nChannels + channel]) - float((uchar)im_src->imageData[(i+naddr[neighbor][0])*im_src->widthStep + (j+naddr[neighbor][1])*im_src->nChannels + channel]);
                            count_neighbors++;
                        }
                    }
                    fp = (sum_f + sum_vpq)/(float)count_neighbors;
                    error = fabs(fp - cvmGet(im_new, i+offset[0], j+offset[1]));
                    if(ok && error > EPS * (1+fabs(fp))){
                        ok = 0;
                    }
                    cvmSet(im_new, i+offset[0], j+offset[1], fp);
                }
            }
        }
        if(ok){
            break;
        }
    }
    
    for(i=0; i<im_dst->height; i++){
        for(j=0; j<im_dst->width; j++){
            if(cvmGet(im_new, i, j) > 255){
                cvmSet(im_new, i, j, 255.0);
            }
            else if(cvmGet(im_new, i, j) < 0){
                cvmSet(im_new, i, j, 0.0);
            }
            im_dst->imageData[(i)*im_dst->widthStep + (j)*im_dst->nChannels + channel] = (uchar)cvmGet(im_new, i, j);
        }
    }
    return 1;
}

int poisson_solver(IplImage *im_src, IplImage *im_dst, IplImage *im_mask, int channel, int *offset){
    int i, j, loop, neighbor, count_neighbors, flag_edge, ok;
    float error, sum_f, sum_fstar, sum_vpq, fp, fq, gp, gq;
    int naddr[NUM_NEIGHBOR][2]={{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    CvMat* im_new = cvCreateMat(im_dst->height, im_dst->width, CV_64F);
    for(i=0; i<im_dst->height; i++){
        for(j=0; j<im_dst->width; j++){
            cvmSet(im_new, i, j, (double)((uchar)im_dst->imageData[(i)*im_dst->widthStep + (j)*im_dst->nChannels + channel]));
        }
    }
    
    for(loop=0; loop<LOOP_MAX; loop++){
        ok=1;
        for(i=0; i<im_mask->height; i++){
            for(j=0; j<im_mask->width; j++){
                if(int((uchar)im_mask->imageData[i*im_mask->widthStep + j]) > 0){
                    sum_f=0.0;
                    sum_fstar=0.0;
                    sum_vpq=0.0;
                    count_neighbors=0;
                    flag_edge=0;
                    for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++){
                        if(int((uchar)im_mask->imageData[(i+naddr[neighbor][0])*im_mask->widthStep + (j+naddr[neighbor][1])]) == 0){
                            flag_edge = 1;
                            break;
                        }
                    }
                    if(flag_edge == 0){
                        for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++){
                            if(i+offset[0]+naddr[neighbor][0] >= 0 && j+offset[1]+naddr[neighbor][1] >= 0 && i+offset[0]+naddr[neighbor][0] < im_dst->height && j+offset[1]+naddr[neighbor][1] < im_dst->width){
                                sum_f += cvmGet(im_new, i+offset[0]+naddr[neighbor][0], j+offset[1]+naddr[neighbor][1]);
                                sum_vpq += float((uchar)im_src->imageData[(i)*im_src->widthStep + (j)*im_src->nChannels + channel]) - float((uchar)im_src->imageData[(i+naddr[neighbor][0])*im_src->widthStep + (j+naddr[neighbor][1])*im_src->nChannels + channel]);
                                count_neighbors++;
                            }
                        }
                    }
                    else{
                        for(neighbor=0; neighbor<NUM_NEIGHBOR; neighbor++){
                            if(i+offset[0]+naddr[neighbor][0] >= 0 && j+offset[1]+naddr[neighbor][1] >= 0 && i+offset[0]+naddr[neighbor][0] < im_dst->height && j+offset[1]+naddr[neighbor][1] < im_dst->width){
                                fp = (double)((uchar)im_dst->imageData[(i+offset[0])*im_dst->widthStep + (j+offset[1])*im_dst->nChannels + channel]);
                                fq = (double)((uchar)im_dst->imageData[(i+offset[0]+naddr[neighbor][0])*im_dst->widthStep + (j+offset[1]+naddr[neighbor][1])*im_dst->nChannels + channel]);
                                gp = (double)((uchar)im_src->imageData[(i)*im_src->widthStep + (j)*im_src->nChannels + channel]);
                                gq = (double)((uchar)im_src->imageData[(i+naddr[neighbor][1])*im_src->widthStep + (j+naddr[neighbor][1])*im_src->nChannels + channel]);
                                sum_fstar += fq;
                                if( fabs(fp-fq) > fabs(gp-gq)){
                                    sum_vpq += fp-fq;
                                }
                                else{
                                    sum_vpq += gp-gq;
                                }
                                count_neighbors++;
                            }
                        }
                    }
                    fp = (sum_f + sum_fstar + sum_vpq)/(float)count_neighbors;
                    error = fabs(fp - cvmGet(im_new, i+offset[0], j+offset[1]));
                    if( ok && error > EPS * (1+fabs(fp))){
                        ok = 0;
                    }
                    cvmSet(im_new, i+offset[0], j+offset[1], fp);
                }
            }
        }
        if(ok){
            break;
        }
    }
    
    for(i=0; i<im_dst->height; i++){
        for(j=0; j<im_dst->width; j++){
            if(cvmGet(im_new, i, j) > 255){
                cvmSet(im_new, i, j, 255.0);
            }
            else if(cvmGet(im_new, i, j) < 0){
                cvmSet(im_new, i, j, 0.0);
            }
            im_dst->imageData[(i)*im_dst->widthStep + (j)*im_dst->nChannels + channel] = (uchar)cvmGet(im_new, i, j);
        }
    }
    return 1;
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

bool check_in_range(int angle,double buf_x,double buf_y){//直線の傾きが指定された角度の範囲内かの判定
    double lim_y=abs(buf_x*cos(angle/180*M_PI));//絶対値で返す
    if (lim_y>abs(buf_y)){
        printf("true ");
        return true;
    }else{
        printf("false ");
        return false;
    }
}
bool is_not_lean(cv::Mat input,cv::Rect roi_rect){//顔の領域と元画像から、傾きが適度な範囲かを判定する
    if(input.empty()){printf("ERROR:resource image not found!\n");exit(0);}
    cv::Mat GrayImg;
    cvtColor(input(roi_rect), GrayImg, CV_BGR2GRAY);
    std::vector<cv::Rect> eye_areas;
    ///目の検出
    // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
    nested_cascade_eye.detectMultiScale(GrayImg, eye_areas,1.1, 3,CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
    printf("eye:%d ",eye_areas.size());
    if(eye_areas.size()==2){
        double buf_x=(eye_areas[1].x+eye_areas[1].width/2)-(eye_areas[2].x+eye_areas[1].width/2);
        double buf_y=(eye_areas[1].x+eye_areas[1].width/2)-(eye_areas[2].x+eye_areas[1].width/2);
        return check_in_range(ANGLE,buf_x,buf_y);
    }else{
        return false;
    }
}

void generate_face(cv::Mat input,cv::Mat dst_img){//in BGR out BGRで書く
    
    int offset[2];
    IplImage *im_src =NULL, *im_dst = NULL, *im_mask = NULL;
    cv::Mat mask_img,not_masked,mask_img_gray;
    std::vector<cv::Rect> faces,faces1,faces2;
    cv::Point center2,center1;
    int radius=0,radius1=0,radius2=0;
    cv::Point p1,p2,p3,p4;
    
    //////loading destination image() & find position and size
    //loading destination file(for face)
    if(dst_img.empty()){printf("ERROR:distination image not found!\n");exit(0);}
    faces1=get_faces(dst_img,scale,cascade);
    printf("deteced faces:%d\n",(int)faces1.size());//DEBUG 1のはず..
    cv::Rect roi_rect2;
    for (int i=0; i<faces1.size(); i++) {
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        center1.x = cv::saturate_cast<int>((faces1[i].x + faces1[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center1.y = cv::saturate_cast<int>((faces1[i].y + faces1[i].height*0.5)*scale);
        radius1 = cv::saturate_cast<int>((faces1[i].width + faces1[i].height)*0.25*scale);
        p3.x=center1.x-radius;p3.y=center1.y-radius;
        p4.x=center1.x+radius;p4.y=center1.y+radius;
        cv::Rect buf_rect(center2.x-radius2,center2.y-radius2,radius2*2,radius2*2);//左上のx座標,y座標,width,depthというふうに格納していく
        roi_rect2=buf_rect;
    }
    
    
    //loading resource file(for face)
    if(input.empty()){printf("ERROR:resource image not found!\n");exit(0);}
    faces2=get_faces(input, scale, cascade);
    //printf("deteced faces:%d\n",(int)faces2.size());//TODO:読み込めない時の処
    //assert(faces2.size()==1);
    for (int i=0; i<1; i++) {
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        //顔全体の矩形の確定
        center2.x = cv::saturate_cast<int>((faces2[i].x + faces2[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center2.y = cv::saturate_cast<int>((faces2[i].y + faces2[i].height*0.5)*scale);
        radius2 = cv::saturate_cast<int>((faces2[i].width + faces2[i].height)*0.25*scale);
        p1.x=center2.x-radius;p1.y=center2.y-radius;
        p2.x=center2.x+radius;p2.y=center2.y+radius;
        cv::Rect roi_rect(center2.x-radius2,center2.y-radius2,radius2*2,radius2*2);//左上のx座標,y座標,width,depthというふうに格納していく
        //find eyes
        //グレースケール画像（目検出の処理はグレースケール画像で行う））
        cv::Mat GrayImg;
        //入力画像をグレースケール画像に変換
        cvtColor(input(roi_rect), GrayImg, CV_BGR2GRAY);
        //目の学習データ（openCVはすでに目のデータを用意している））//TODO 画像サイズ縮小などによる高速化
        std::vector<cv::Rect> eye_areas,nose_areas,mouth_areas;
        ///目の検出
        // 画像，出力矩形，縮小スケール，最低矩形数，（フラグ），最小矩形
        nested_cascade_eye.detectMultiScale(GrayImg, eye_areas,1.1, 4,CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
        nested_cascade_nose.detectMultiScale(GrayImg, nose_areas,1.1, 3,CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
        nested_cascade_mouth.detectMultiScale(GrayImg, mouth_areas,1.1, 3,CV_HAAR_SCALE_IMAGE,cv::Size(10,10));
        // 最初のイテレータから、最後のイテレータになるまで、イテレータを一つずつ先に進めます。
        mask_img.create(input.size(), CV_8UC3);
        mask_img=cv::Scalar(0,0,0);//真っ黒に
        cv::Rect iElement;
        for(cv::vector<cv::Rect>::iterator itiElement = eye_areas.begin(); itiElement != eye_areas.end(); ++itiElement) {
            iElement = *itiElement;      // itiElement を使って繰り返し、配列要素を扱う
            cv::Rect buf;//元々のマスク位置の部分の足し算
            //複数化への対応はできていない
            //buf.x=iElement.x+roi_rect.x-iElement.height*(PARTS_RATE-1)/2;//元々の矩形の座標を忘れずに
            buf.x=iElement.x+roi_rect.x;
            buf.y=iElement.y+roi_rect.y;
            buf.width=iElement.width;
            buf.height=iElement.height;
            //            buf.height=iElement.height*PARTS_RATE;
            not_masked=mask_img(buf);
            not_masked=cv::Scalar(255,255,255);//真っ白に
        }
        for(cv::vector<cv::Rect>::iterator itiElement = nose_areas.begin(); itiElement != nose_areas.end(); ++itiElement) {
            iElement = *itiElement;      // itiElement を使って繰り返し、配列要素を扱う
            cv::Rect buf;//元々のマスク位置の部分の足し算
            //複数化への対応はできていない
            buf.x=iElement.x+roi_rect.x-(PARTS_RATE-1)/2*iElement.height;//元々の矩形の座標を忘れずに
            buf.y=iElement.y+roi_rect.y;
            buf.width=iElement.width;
            buf.height=iElement.height*PARTS_RATE;
            not_masked=mask_img(buf);
            not_masked=cv::Scalar(255,255,255);//真っ白に
        }
        for(cv::vector<cv::Rect>::iterator itiElement = mouth_areas.begin(); itiElement != mouth_areas.end(); ++itiElement) {
            iElement = *itiElement;      // itiElement を使って繰り返し、配列要素を扱う
            cv::Rect buf;//元々のマスク位置の部分の足し算
            //複数化への対応はできていない
            buf.x=iElement.x+roi_rect.x;//元々の矩形の座標を忘れずに
            buf.y=iElement.y+roi_rect.y-iElement.height*PARTS_RATE;
            buf.width=iElement.width;
            buf.height=iElement.height*2*PARTS_RATE;
            not_masked=mask_img(buf);
            not_masked=cv::Scalar(255,255,255);//真っ白に
        }
        cvtColor(mask_img,mask_img_gray,CV_RGB2GRAY);
        input(roi_rect).copyTo(not_masked);
        //        cv::namedWindow("input",1);
        //        cv::imshow("input",mask_img);
    }
    double ratio=(double)radius2/(double)radius1;//図形1:dst_imgを図形2:resource側へ合わせる
    int difx=center1.x*ratio-center2.x;
    int dify=center1.y*ratio-center2.y;
    //printf("%f%d%d\n",ratio,difx,dify);
    cv::Mat expanded_output(cv::saturate_cast<int>(dst_img.rows*ratio),cv::saturate_cast<int>(dst_img.cols*ratio),CV_8UC1);
    cv::resize(dst_img,expanded_output,expanded_output.size(),0,0,cv::INTER_LINEAR);
    //resizing source image & calc offset
    //ref:http://bicycle.life.coocan.jp/takamints/index.php/doc/opencv/doc/Mat_conversion//
    //http://d.hatena.ne.jp/kamekamekame877/20110621
    IplImage buf1=input;//特殊なコピーコンストラクタが呼ばれてるかららしい
    IplImage buf2=mask_img_gray;//同じく
    cv::medianBlur(expanded_output(roi_rect2),expanded_output(roi_rect2),33);
    IplImage buf3=expanded_output;
    //cv::namedWindow("input",1);
    //cv::imshow("input",input);
    im_src=&buf1;
    im_mask=&buf2;
    im_dst=&buf3;
    
    offset[0]=dify;
    offset[1]=difx;
    
    for(int i=0;i<3;i++){// i:channnels
        quasi_poisson_solver(im_src, im_dst, im_mask, i, offset);
        //poisson_solver(im_src, im_dst, im_mask, i, offset);
    }
    ///http://bicycle.life.coocan.jp/takamints/index.php/doc/opencv/doc/Mat_conversion を丸写し
    //    cv::Mat mat(im_dst);  //(1)コピーコンストラクタ
    //    cv::Mat mat2;
    //    mat2 = im_dst;        //(2)ポインタの代入
    cv::Mat mat2 = cv::cvarrToMat(im_dst);
    cv::Mat hoge;
    hoge.create(TEXTURE_HEIGHT, TEXTURE_WIDTH, CV_8UC3);
    cv::resize(mat2, hoge,hoge.size());
    generated=hoge.clone();
    //後始末について
    //cvReleaseImage(&im_src);
    //cvReleaseImage(&im_dst);
    //cvReleaseImage(&im_mask);
    return;
}
///////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////from index_finger_detector.cpp
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

void get_rect(cv::Point center,int radius,cv::Point *p1,cv::Point *p2,double magnification){
    p1->x=(center.x-radius*magnification>=0)? center.x-radius*magnification : 0 ;
    p1->y=(center.y-radius*magnification>=0)? center.y-radius*magnification : 0 ;
    p2->x=(center.x+radius*magnification<=VIDEO_WIDTH)? center.x+radius*magnification : VIDEO_WIDTH ;
    p2->y=(center.y+radius*magnification<=VIDEO_WIDTH)? center.y+radius*magnification : VIDEO_WIDTH ;
    return;
}

/////////////////////////////////////////////////////////////////////////////////////////////

void draw_string(double x0,double y0,std::string str){
    //glColor3d(1.0, 0.0, 0.0);
    glRasterPos2f(x0, y0);
    int size = (int)str.size();
    for(int i = 0; i < size; ++i){
        char ic = str[i];
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ic);
    }
    glColor3d(1.0, 1.0, 1.0);
}

void set_texture(){//reading textures
    //initializing
    for(int i=0; i<4; i++){// destination images
        cv::Mat input = cv::imread(inputFileNames[i],1);//
        if (input.empty()) {
            printf("Can't read:%s",inputFileNames[i]);
            exit(0);
        }
        cv::Mat buf;
        buf.create(TEXTURE_HEIGHT*0.9, TEXTURE_WIDTH*0.9, CV_8UC3);
        cv::resize(input, buf, buf.size());
        cv::cvtColor(buf, buf, CV_BGR2RGB);
        resources.push_back(buf);
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[i]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - buf.cols)/2, (TEXTURE_HEIGHT- buf.rows)/2, buf.cols, buf.rows, GL_RGB, GL_UNSIGNED_BYTE, buf.data);
        
        //cv::imshow("input_img",input);
        //cv::waitKey(1000);
    }
}
void draw_captured_photo(){//いわゆる右上の、動画と静止画が切り替わるゾーンの描画
    //描画
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[1]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[2]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[5]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[4]);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    if (g_display_mode==9){//目安としての平行線の描画
        glColor3d( 1.0, 1.0, 1.0);
        glBegin( GL_LINE_LOOP );
        glVertex2d(0.3,0.5);
        glVertex2d(0.7,0.5);
        glEnd();
    }
}
void draw_destination_photo(){//texturehandle5に登録されているものを左上に描画する関数.generatedなのかselectedなのか、どっちを選択するかはきっちり決めないと
    glEnable(GL_TEXTURE_2D);
    if (isgenerated) {
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[5]);
    }else{
        if (selected_photo==0) {
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[0]);
        }else{
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[selected_photo-1]);//photo 1 -handle 0 に対応
        }
    }
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[0]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[1]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[4]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[3]);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void draw_initial_photo(){
    glEnable(GL_TEXTURE_2D);
    
    //fig1
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[0]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[6]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[7]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[12]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[11]);
    glEnd();
    
    //fig2
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[1]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[7]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[8]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[13]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[12]);
    glEnd();
    
    //fig3
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[2]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[8]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[9]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[14]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[13]);
    glEnd();
    
    //fig4
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[3]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);glVertex2dv(vertices[9]);
    glTexCoord2d(1.0, 0.0);glVertex2dv(vertices[10]);
    glTexCoord2d(1.0, 1.0);glVertex2dv(vertices[15]);
    glTexCoord2d(0.0, 1.0);glVertex2dv(vertices[14]);
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}

void init_GL(int argc, char *argv[]){
    glutInit(&argc,argv);// OpenGLの初期化
    glutInitDisplayMode(GLUT_RGBA);// ディスプレイモードをRGBAモードに設定　多分最後のAは透過色用
    glutInitWindowSize(WINDOW_X, WINDOW_Y);//ウィンドウサイズを指定
    glutCreateWindow(WINDOW_NAME);// ウィンドウを「生成」。まだ「表示」はされない。
}
void init(){
    glClearColor(0.0, 0.0,0.0, 0.0);// 背景の塗りつぶし色を指定　この場合ただの真っ黒
    // set texture property
    glGenTextures(6, g_TextureHandles);
    for(int i = 0; i < 6; i++){
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[i]);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEXTURE_WIDTH,TEXTURE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
    set_texture();
    
    // loading classifier
    std::string cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    std::string nested_cascadeName_eye = "/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_eye.xml";
    std::string nested_cascadeName_nose= "/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml";
    std::string nested_cascadeName_mouth= "/usr/local/Cellar/opencv/2.4.9/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";
    if((!nested_cascade_eye.load(nested_cascadeName_eye))||(!nested_cascade_nose.load(nested_cascadeName_nose))||(!nested_cascade_mouth.load(nested_cascadeName_mouth))){
        printf("Error:can't open resource\n");exit(0);
    }
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadefile見つからん！\n");exit(0);
    }
}

// キーボードに変化があった時に呼び出されるコールバック関数。
void glut_keyboard(unsigned char key, int x, int y){
    switch(key){
        case 'q':
        case 'Q':
        case '\033': // Escキーのこと
            exit(0);
        case '1':
            if (g_display_mode==7){g_display_mode=6;isgenerated=false;}else{g_display_mode = 2;}
            selected_photo = 1;
            break;
        case '2':
            if (g_display_mode==7){g_display_mode=6;isgenerated=false;}else{g_display_mode = 2;}
            selected_photo = 2;
            break;
        case '3':
            if (g_display_mode==7){g_display_mode=6;isgenerated=false;}else{g_display_mode = 2;}
            selected_photo = 3;
            break;
        case '4':
            if (g_display_mode==7){g_display_mode=6;isgenerated=false;}else{g_display_mode = 2;}
            selected_photo = 4;
            break;
        case 'm':
            g_display_mode = 3;
            break;
        case 'a':
            g_display_mode = 8;
            break;
        case 'g':
            g_display_mode = 6;
            break;
        case 'n':
            g_display_mode = 0;
            isgenerated = false;
            break;
    }
    
    glutPostRedisplay(); // 「ディスプレイのコールバック関数を呼んで」と指示する。描画のためのフラグ設定をしているにすぎない
}
// ディスプレイに変化があった時に呼び出されるコールバック関数。
// 「ディスプレイに変化があった時」は、glutPostRedisplay() で指示する。→ここに書いとけば基本的に反映されるはず
void glut_display(){
    glClear(GL_COLOR_BUFFER_BIT); // 今まで画面に描かれていたものを消す
    switch(g_display_mode){
        case 0:{
            draw_string(-1, -0.35,str1);
            draw_string(-1, -0.45,str2);
            //再起動の時にはここまで戻ってこないとダメなはず
            selected_photo=0;
            isprinted=false;
            iscaptured=false;
            break;}
        case 2:
            draw_string(-1, -0.35,str3);
            break;
        case 3:
            cap.open(0);
            cap.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);
            if (!cap.isOpened()){printf("Can't open video input\n");exit(0);}
            g_display_mode=4;//captureはopenしてる。いわゆる録画モード。
            break;
        case 4:
            draw_string(-1, -0.35,str4);
            break;
        case 5:
            draw_string(-1, -0.35,str5);
            draw_string(-1, -0.45,str6);
            break;
        case 6:
            draw_string(-1, -0.35,str7);
            break;
        case 7:
            draw_string(-1, -0.27,str8);
            draw_string(-1, -0.35,str9);
            draw_string(-1, -0.42,str10);
            break;
        case 8:
            cap.open(0);
            cap.set(CV_CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT);
            if (!cap.isOpened()){printf("Can't open video input\n");exit(0);}
            g_display_mode=9;//captureはopenしてる。いわゆる録画モード。
            break;
        case 9:
            draw_string(-1, -0.35, str11);
            break;
        default:
            break;
    }
    draw_captured_photo();
    draw_initial_photo();
    draw_destination_photo();
    glFlush(); // ここで画面に描画をする
}

void glut_idle(){
    if(isgenerated==true){
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[5]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - generated.cols)/2, (TEXTURE_HEIGHT- generated.rows)/2, generated.cols,generated.rows, GL_RGB, GL_UNSIGNED_BYTE,generated.data);
    }else if (iscaptured && !(g_display_mode==4)&& !(g_display_mode==9)&& !isprinted){//顔がすでに取れていて、まだテクスチャ貼り付けできてなくて、かつ、ビデオが開いていない時
        //色空間はRGBであること！！
        printf("have captured face and cap is not opened\n");
        //same as set_texture()
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - captured.cols)/2, (TEXTURE_HEIGHT- captured.rows)/2, captured.cols, captured.rows, GL_RGB, GL_UNSIGNED_BYTE, captured.data);
        isprinted=true;
    }else if(g_display_mode==4){//ビデオが開いてる時
        isprinted=false;
        iscaptured=false;
        //変数宣言とビデオ入力(globalにしなくて大丈夫だよね？..)
        cv::Mat hsv_skin_img= cv::Mat(cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT),CV_8UC3);
        cv::Mat hsv_img,gray_img,bin_img,input_img,copy_input_img,smooth_img,dst_img,dst_img_norm;
        cap >> input_img;
        cap >> copy_input_img;
        if (input_img.empty()) {printf("window empty!\n");exit(0);}
        // 手の検出のための前処理
        hsv_skin_img = cv::Scalar(0,0,0);
        cv::medianBlur(input_img,smooth_img,7);	//eliminate noises
        cv::cvtColor(smooth_img,hsv_img,CV_BGR2HSV);// convert(RGB→HSV)
        for(int y=0; y<VIDEO_HEIGHT;y++){
            for(int x=0; x<VIDEO_WIDTH; x++){
                int a = (int)(hsv_img.step*y+(x*3));
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
        if (contours.size()==0) {return;}//if no contour
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
        cv::waitKey(100);
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
            cv::circle(input_img,detected_point,20,cv::Scalar(255,255,0),-1,8,0);
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
                get_rect(center,radius,&p1,&p2,MAGNIFICATION);
                cv::Rect roi_rect(p1,p2);
                //cv::rectangle(input_img, roi_rect, cv::Scalar(255,0,0));
                g_display_mode=5;
                iscaptured=true;
                cap.release();
                cv::Mat buf;
                buf.create(TEXTURE_HEIGHT*0.9, TEXTURE_WIDTH*0.9, CV_8UC3);
                cv::resize(copy_input_img(roi_rect), buf, buf.size());
                cv::cvtColor(buf, buf, CV_BGR2RGB);
                captured=buf.clone();
                input_img=cv::Scalar(0,0,0);
                glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
                glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - captured.cols)/2, (TEXTURE_HEIGHT- captured.rows)/2, captured.cols, captured.rows, GL_RGB, GL_UNSIGNED_BYTE, captured.data);
            }
            
            
        }
        //最後の描画処理
        if (!iscaptured) {
            cv::cvtColor(input_img,input_img, CV_BGR2RGB);
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - input_img.cols)/2, (TEXTURE_HEIGHT- input_img.rows)/2, input_img.cols, input_img.rows, GL_RGB, GL_UNSIGNED_BYTE, input_img.data);
        }else{
            input_img=cv::Scalar(0,0,0);
            cv::cvtColor(input_img,input_img, CV_BGR2RGB);
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - input_img.cols)/2, (TEXTURE_HEIGHT- input_img.rows)/2, input_img.cols, input_img.
                            rows, GL_RGB, GL_UNSIGNED_BYTE, input_img.data);
        }
        
    }else if (g_display_mode==9){
        isprinted=false;
        iscaptured=false;
        cv::Mat input_img;
        cap >> input_img;
        std::vector<cv::Rect> faces=get_faces(input_img,scale,cascade);
        if (faces.size()>=1){
            //printf("face detected(auto mode)");
            cv::Point center,p1,p2;int radius;
            //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
            center.x = cv::saturate_cast<int>((faces[0].x + faces[0].width*0.5)*scale);//scaleはここで戻していることに注意！
            center.y = cv::saturate_cast<int>((faces[0].y + faces[0].height*0.5)*scale);
            radius = cv::saturate_cast<int>((faces[0].width + faces[0].height)*0.25*scale);
            get_rect(center,radius,&p1,&p2,1.0);
            cv::Rect roi_rect(p1,p2);
            //cv::rectangle(input_img, roi_rect,cv::Scalar(255,0,0));
            //            if (is_not_lean(input_img, roi_rect)) {//目の傾きが大丈夫なら
            if(true){
                get_rect(center,radius,&p1,&p2,MAGNIFICATION);
                cv::Rect roi_rect1(p1,p2);
                //cv::rectangle(input_img, roi_rect1, cv::Scalar(255,0,0));
                cv::Mat buf;
                buf.create(TEXTURE_HEIGHT, TEXTURE_WIDTH, CV_8UC3);
                cv::resize(input_img(roi_rect1), buf, buf.size());
                
                g_display_mode=5;
                iscaptured=true;
                cap.release();
                cv::cvtColor(buf,buf,CV_BGR2RGB);
                captured=buf.clone();
                input_img=cv::Scalar(0,0,0);
                glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
                glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - captured.cols)/2, (TEXTURE_HEIGHT- captured.rows)/2, captured.cols,captured.rows, GL_RGB, GL_UNSIGNED_BYTE,captured.data);
            }
        }else{
            cv::cvtColor(input_img,input_img, CV_BGR2RGB);
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[4]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - input_img.cols)/2, (TEXTURE_HEIGHT- input_img.rows)/2, input_img.cols, input_img.rows, GL_RGB, GL_UNSIGNED_BYTE, input_img.data);
        }
    }else if (g_display_mode==6){
        cv::cvtColor(captured,captured,CV_RGB2BGR);
        cv::cvtColor(resources[selected_photo-1],resources[selected_photo-1],CV_RGB2BGR);
        //合成
        generate_face(captured,resources[selected_photo-1]);
        //できたテクスチャの登録
        cv::cvtColor(generated,generated,CV_BGR2RGB);
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[5]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - generated.cols)/2, (TEXTURE_HEIGHT- generated.rows)/2, generated.cols, generated.rows, GL_RGB, GL_UNSIGNED_BYTE,generated.data);
        isgenerated=true;
        //        cv::namedWindow("generated",1);
        //        cv::imshow("generated",generated);
        g_display_mode=7;
        cv::cvtColor(resources[selected_photo-1],resources[selected_photo-1],CV_RGB2BGR);
    }
    glutPostRedisplay();
}
void set_callback_functions(){//この関数内で指定したものを行わないと、MainLoopが終了しない
    glutDisplayFunc(glut_display);// ディスプレイに変化があった時に呼ばれるコールバック関数を登録
    glutKeyboardFunc(glut_keyboard);// キーボードに変化があった時に呼び出されるコールバック関数を登録
    glutIdleFunc(glut_idle);//何もない時の監視用
}

int main(int argc, char *argv[]){
    /* OpenGLの初期化 */
    init_GL(argc,argv);
    
    /* このプログラム特有の初期化 */
    init();
    
    /* コールバック関数の登録 */
    set_callback_functions();
    
    /* メインループ */
    glutMainLoop();// 無限ループ。コールバック関数が呼ばれるまでずっと実行される。
    
    return 0;
}
