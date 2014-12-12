//gl_mainwindow.cpp
//include二行と、build_targetをOSx.10.8以下,後はpathにOPENGLとGLUTを入れればOK
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cv.h>
#include <highgui.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#define WINDOW_X (640)
#define WINDOW_Y (640)
#define WINDOW_NAME "facegenerator" //画像については、600*600以上が読み込めなくなった
#define TEXTURE_HEIGHT 600
#define TEXTURE_WIDTH 600


//global variable
int g_display_mode = 9;
GLuint g_TextureHandles[6] = {1,2,3,4,5,6};
cv::VideoCapture cap;
//http://slis.tsukuba.ac.jp/~fujisawa.makoto.fu/lecture/iml/text/screen_character.html writing characters on OpenGL
std::string str1="choose destination image.";
std::string str2="press 1,2,3,4 to choose one, q to quit";
std::string str3="press c to capture face";
std::string str4="face captured";
std::string str5="press g to generate new face,c to capture again";
std::string str6="face generated!";

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

void init_GL(int argc, char *argv[]){
    glutInit(&argc,argv);// OpenGLの初期化
    glutInitDisplayMode(GLUT_RGBA);// ディスプレイモードをRGBAモードに設定　多分最後のAは透過色用
    glutInitWindowSize(WINDOW_X, WINDOW_Y);//ウィンドウサイズを指定
    glutCreateWindow(WINDOW_NAME);// ウィンドウを「生成」。まだ「表示」はされない。
}

void set_texture(){//reading textures
    const char* inputFileNames[4] = {"/Users/naoto/git/opencv_gl/opencv/mona_lisa.png", "/Users/naoto/git/opencv_gl/opencv/kawagoe.jpg", "/Users/naoto/git/opencv_gl/opencv/horikita.jpg","/Users/naoto/git/opencv_gl/opencv/fukuyama.jpg"};
    //initializing
    for(int i=0; i<4; i++){// destination images
        cv::Mat input = cv::imread(inputFileNames[i],1);// convert BGR -> RGB
        if (input.empty()) {
            printf("Can't read:%s",inputFileNames[i]);
            exit(0);
        }
        cv::Mat buf;
        buf.create(TEXTURE_HEIGHT*0.9, TEXTURE_WIDTH*0.9, CV_8UC3);
        cv::resize(input, buf, buf.size());
        cv::cvtColor(buf, buf, CV_BGR2RGB);
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[i]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - buf.cols)/2, (TEXTURE_HEIGHT- buf.rows)/2, buf.cols, buf.rows, GL_RGB, GL_UNSIGNED_BYTE, buf.data);
        //cv::imshow("input_img",input);
        //cv::waitKey(1000);
    }
}


void draw_initial_photo(){
    glEnable(GL_TEXTURE_2D);
    
    //fig1
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[0]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);
    glVertex2dv(vertices[6]);
    glTexCoord2d(1.0, 0.0);
    glVertex2dv(vertices[7]);
    glTexCoord2d(1.0, 1.0);
    glVertex2dv(vertices[12]);
    glTexCoord2d(0.0, 1.0);
    glVertex2dv(vertices[11]);
    glEnd();
    
    //fig2
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[1]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);
    glVertex2dv(vertices[7]);
    glTexCoord2d(1.0, 0.0);
    glVertex2dv(vertices[8]);
    glTexCoord2d(1.0, 1.0);
    glVertex2dv(vertices[13]);
    glTexCoord2d(0.0, 1.0);
    glVertex2dv(vertices[12]);
    glEnd();
    
    //fig3
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[2]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);
    glVertex2dv(vertices[8]);
    glTexCoord2d(1.0, 0.0);
    glVertex2dv(vertices[9]);
    glTexCoord2d(1.0, 1.0);
    glVertex2dv(vertices[14]);
    glTexCoord2d(0.0, 1.0);
    glVertex2dv(vertices[13]);
    glEnd();
    
    //fig4
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[3]);
    //座標系はyが下向きに取られていることに注意
    glBegin(GL_POLYGON);
    glTexCoord2d(0.0, 0.0);
    glVertex2dv(vertices[9]);
    glTexCoord2d(1.0, 0.0);
    glVertex2dv(vertices[10]);
    glTexCoord2d(1.0, 1.0);
    glVertex2dv(vertices[15]);
    glTexCoord2d(0.0, 1.0);
    glVertex2dv(vertices[14]);
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}

void init(){
    glClearColor(0.0, 0.0,0.0, 0.0);// 背景の塗りつぶし色を指定　この場合ただの真っ黒
    // set texture property
    glGenTextures(6, g_TextureHandles);
    for(int i = 0; i < 4; i++){
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[i]);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEXTURE_WIDTH,TEXTURE_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
    set_texture();
}

// キーボードに変化があった時に呼び出されるコールバック関数。
void glut_keyboard(unsigned char key, int x, int y){
    switch(key){
            
        case 'q':
        case 'Q':
        case '\033': // Escキーのこと
            exit(0);
        case '1':
            g_display_mode = 0;
            break;
        case '2':
            g_display_mode = 1;
            break;
        case '3':
            g_display_mode = 2;
            break;
        case '4':
            g_display_mode = 3;
    }
    
    glutPostRedisplay(); // 「ディスプレイのコールバック関数を呼んで」と指示する。描画のためのフラグ設定をしているにすぎない
}
// ディスプレイに変化があった時に呼び出されるコールバック関数。
// 「ディスプレイに変化があった時」は、glutPostRedisplay() で指示する。
void glut_display(){
    glClear(GL_COLOR_BUFFER_BIT); // 今まで画面に描かれていたものを消す
    
    switch(g_display_mode){
        case 0:
        case 1:
        case 2:
        case 3:
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, g_TextureHandles[g_display_mode]);
            //座標系はyが下向きに取られていることに注意
            glBegin(GL_POLYGON);
            glTexCoord2d(0.0, 0.0);
            glVertex2dv(vertices[0]);
            glTexCoord2d(1.0, 0.0);
            glVertex2dv(vertices[1]);
            glTexCoord2d(1.0, 1.0);
            glVertex2dv(vertices[4]);
            glTexCoord2d(0.0, 1.0);
            glVertex2dv(vertices[3]);
            glEnd();
            
            glDisable(GL_TEXTURE_2D);
            break;
        case 9:{
            draw_string(-1, -0.35,str1);
            draw_string(-1, -0.45,str2);
            break;}
        case 'c':
            cap.open(0);
            if (!cap.isOpened()){printf("Can't open video input\n");exit(0);}
            
            break;
        case 'r':
            break;
        default:
            break;
    }
    draw_initial_photo();
    glFlush(); // ここで画面に描画をする
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
