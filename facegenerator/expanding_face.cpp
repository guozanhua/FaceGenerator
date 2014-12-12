#include <stdio.h>
#include <cv.h>
#include <highgui.h>

#define LOOP_MAX 10000
#define EPS 2.2204e-016
#define NUM_NEIGHBOR 4

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

// usage: poisson.exe [source image (color)] [destination image (color] [mask image (gray)] [outout image (color] [offset y] [offset x]
int main(int argc, char** argv){
    int offset[2];
    IplImage *im_src =NULL, *im_dst = NULL, *im_mask = NULL;
    cv::Mat input,dst_img,mask_img,not_masked,mask_img_gray;
    
    if(argc != 4){
        fprintf(stderr,"usage: poisson.exe [source image (color)] [destination image (color)] [outout image (color)] [offset y] [offset x]\n");
        exit(0);
    }
    //////loading source image(cv::Mat input) & creating mask(cv::Mat mask_img)
    //loading haar classifier
    std::string cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadefile見つからん！\n");
        return -1;
    }
    //////loading destination image() & find position and size
    //loading destination file(for face)
    dst_img=cv::imread(argv[2],1);
    if(dst_img.empty()){
        printf("ERROR:distination image not found!\n");
        return 0;
    }
    //preprocessing
    double scale = 4.0;
    cv::Mat gray1, smallImg1(cv::saturate_cast<int>(dst_img.rows/scale),cv::saturate_cast<int>(dst_img.cols/scale),CV_8UC1);
    cv::cvtColor(dst_img, gray1, CV_BGR2GRAY);
    cv::resize(gray1, smallImg1, smallImg1.size(),0,0,cv::INTER_LINEAR);
    cv::equalizeHist(smallImg1, smallImg1);//ヒストグラムビンの合計値が 255 になるようヒストグラムを正規化
    std::vector<cv::Rect> faces,faces1,faces2;
    cv::Point center2,center1;
    int radius,radius1,radius2;
    cv::Point p1,p2;
    //find face size
    cascade.detectMultiScale(smallImg1, faces1,1.1,2,CV_HAAR_SCALE_IMAGE,cv::Size(20,20));
    printf("deteced faces:%d\n",(int)faces1.size());
    for (int i=0; i<faces1.size(); i++) {
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        center1.x = cv::saturate_cast<int>((faces1[i].x + faces1[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center1.y = cv::saturate_cast<int>((faces1[i].y + faces1[i].height*0.5)*scale);
        radius1 = cv::saturate_cast<int>((faces1[i].width + faces1[i].height)*0.25*scale);
    }
    
    //loading resource file(for face)
    input=cv::imread(argv[1],1);
    if(input.empty()){
        printf("ERROR:resource image not found!\n");
        return 0;
    }
    //preprocessing
    cv::Mat gray, smallImg(cv::saturate_cast<int>(input.rows/scale),cv::saturate_cast<int>(input.cols/scale),CV_8UC1);
    cv::cvtColor(input, gray, CV_BGR2GRAY);
    cv::resize(gray, smallImg, smallImg.size(),0,0,cv::INTER_LINEAR);
    cv::equalizeHist(smallImg, smallImg);//ヒストグラムビンの合計値が 255 になるようヒストグラムを正規化
    //find face size and generating mask
    cascade.detectMultiScale(smallImg, faces2,1.1,2,CV_HAAR_SCALE_IMAGE,cv::Size(20,20));
    printf("deteced faces:%d\n",(int)faces2.size());
    for (int i=0; i<1; i++) {
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        center2.x = cv::saturate_cast<int>((faces2[i].x + faces2[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center2.y = cv::saturate_cast<int>((faces2[i].y + faces2[i].height*0.5)*scale);
        radius2 = cv::saturate_cast<int>((faces2[i].width + faces2[i].height)*0.25*scale);
        p1.x=center2.x-radius;p1.y=center2.y-radius;
        p2.x=center2.x+radius;p2.y=center2.y+radius;
        cv::Rect roi_rect(center2.x-radius2,center2.y-radius2,radius2*2,radius2*2);//左上のx座標,y座標,width,depthというふうに格納していく
        mask_img.create(input.size(), CV_8UC3);
        mask_img=cv::Scalar(0,0,0);//真っ黒に
        not_masked=mask_img(roi_rect);
        not_masked=cv::Scalar(255,255,255);//真っ白に
        cvtColor(mask_img,mask_img_gray,CV_RGB2GRAY);
        input(roi_rect).copyTo(not_masked);
    }
    double ratio=(double)radius2/(double)radius1;//
    int difx=center1.x*ratio-center2.x;
    int dify=center1.y*ratio-center2.y;
    printf("%f%d%d\n",ratio,difx,dify);
    cv::Mat expanded_output(cv::saturate_cast<int>(dst_img.rows*ratio),cv::saturate_cast<int>(dst_img.cols*ratio),CV_8UC1);
    cv::resize(dst_img,expanded_output,expanded_output.size(),0,0,cv::INTER_LINEAR);
    //resizing source image & calc offset
    
    //    //for debug(checking masks)
    //        cv::namedWindow("input",1);
    //        cv::namedWindow("result",1);
    //        cv::namedWindow("masked",1);
    //        cv::imshow("input", input);
    //        cv::imshow("result", expanded_output);
    //        cv::imshow("masked", mask_img);
    //ref:http://bicycle.life.coocan.jp/takamints/index.php/doc/opencv/doc/Mat_conversion//
    //http://d.hatena.ne.jp/kamekamekame877/20110621
    IplImage buf1=input;//特殊なコピーコンストラクタが呼ばれてるかららしい
    IplImage buf2=mask_img_gray;//同じく
    IplImage buf3=expanded_output;
    im_src=&buf1;
    im_mask=&buf2;
    im_dst=&buf3;
    
    offset[0]=dify;
    offset[1]=difx;
    
    for(int i=0;i<3;i++){// i:channnels
        quasi_poisson_solver(im_src, im_dst, im_mask, i, offset);
        //poisson_solver(im_src, im_dst, im_mask, i, offset);
    }
    cvSaveImage(argv[3],im_dst);
    //    cvReleaseImage(&im_src);
    //    cvReleaseImage(&im_dst);
    //    cvReleaseImage(&im_mask);
    return 0;
}
