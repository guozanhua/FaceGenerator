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
    int i;
    int offset[2];
    IplImage *im_src = NULL, *im_dst = NULL, *im_mask = NULL;
    
    if(argc != 7){
        fprintf(stderr,"usage: poisson.exe [source image (color)] [destination image (color)] [mask image (gray)] [outout image (color)] [offset y] [offset x]\n");
        exit(0);
    }
//////loading source image(cv::Mat input) & creating mask(cv::Mat mask_img)
    cv::Mat input,mask_img,not_masked;
    
    //loading haar classifier
    std::string cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
    cv::CascadeClassifier cascade;
    if(!cascade.load(cascadeName)){
        printf("ERROR: cascadefile見つからん！\n");
        return -1;
    }
    
    //loading resource file(for face)
    input=cv::imread(argv[1],1);
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
        cv::Point center;
        int radius;
        //saturate_castについては　http://opencv.jp/opencv-2svn/cpp/operations_on_arrays.html
        center.x = cv::saturate_cast<int>((faces[i].x + faces[i].width*0.5)*scale);//scaleはここで戻していることに注意！
        center.y = cv::saturate_cast<int>((faces[i].y + faces[i].height*0.5)*scale);
        radius = cv::saturate_cast<int>((faces[i].width + faces[i].height)*0.25*scale);
        cv::Point p1,p2;
        p1.x=center.x-radius;p1.y=center.y-radius;
        p2.x=center.x+radius;p2.y=center.y+radius;
        cv::Rect roi_rect(center.x-radius,center.y-radius,radius*2,radius*2);//左上のx座標,y座標,width,depthというふうに格納していく
        mask_img.create(input.size(), CV_8UC1);
        mask_img=cv::Scalar(0,0,0);
        not_masked=mask_img(roi_rect);
        not_masked=cv::Scalar(255,255,255);
    }
    // for debug(checking masks)
//    cv::namedWindow("result",1);
//    cv::namedWindow("masked",1);
//    cv::imshow("result", input);
//    cv::imshow("masked", mask_img);
//    cv::waitKey(0);
    
    //loading destination image() & find position and size
    //resizing source image & calc offset
    if( (im_src = cvLoadImage( argv[1], CV_LOAD_IMAGE_COLOR)) == 0 ){
        fprintf(stderr,"No such file %s", argv[1]);
        exit(0);
    }
    if( (im_dst = cvLoadImage( argv[2], CV_LOAD_IMAGE_COLOR)) == 0 ){
        fprintf(stderr,"No such file %s", argv[2]);
        exit(0);
    }
    if( (im_mask = cvLoadImage( argv[3], CV_LOAD_IMAGE_GRAYSCALE)) == 0 ){
        fprintf(stderr,"No such file %s", argv[3]);
        exit(0);
    }
    offset[0]=atoi(argv[5]);
    offset[1]=atoi(argv[6]);
    
    for(i=0;i<3;i++){// i:channnels
        quasi_poisson_solver(im_src, im_dst, im_mask, i, offset);
        //poisson_solver(im_src, im_dst, im_mask, i, offset);
    }
    cvSaveImage(argv[4],im_dst);
    cvReleaseImage(&im_src);
    cvReleaseImage(&im_dst);
    cvReleaseImage(&im_mask);
    return 0;
}