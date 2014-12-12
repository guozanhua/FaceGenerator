#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define WINDOW_X (500)
#define WINDOW_Y (500)
#define WINDOW_NAME "test5"
#define TEXTURE_HEIGHT (960)// ここの大きさがめちゃくちゃ大事！
#define TEXTURE_WIDTH (1280)//

void init_GL(int argc, char *argv[]);
void init();
void set_callback_functions();

void glut_display();
void glut_keyboard(unsigned char key, int x, int y);
void glut_mouse(int button, int state, int x, int y);
void glut_motion(int x, int y);
void glut_idle();
void draw_pyramid();
void set_texture();

// グローバル変数
double g_angle1 = 0.0;
double g_angle2 = -3.141592/6;
double g_distance = 10.0;
bool g_isLeftButtonOn = false;
bool g_isRightButtonOn = false;
GLuint g_TextureHandles[1] = {0};
cv::VideoCapture cap;

int main(int argc, char *argv[]){
    /* OpenGLの初期化 */
    init_GL(argc,argv);
    /* コールバック関数の登録 */
    set_callback_functions();
    
    /* このプログラム特有の初期化 */
    init();
    
    
    /* メインループ */
    glutMainLoop();
    
    return 0;
}

void init_GL(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(WINDOW_X,WINDOW_Y);
    glutCreateWindow(WINDOW_NAME);
}

void init(){
    cap.open(0);
    printf("width:%f,height:%f",cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_WIDTH));
    glClearColor(0.2, 0.2, 0.2, 0.2);
    glGenTextures(1, g_TextureHandles);
    glBindTexture(GL_TEXTURE_2D, g_TextureHandles[0]);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEXTURE_WIDTH,TEXTURE_HEIGHT, 0, GL_RGB,GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    
}

void set_callback_functions(){
    glutDisplayFunc(glut_display);
    glutKeyboardFunc(glut_keyboard);
    glutMouseFunc(glut_mouse);
    glutMotionFunc(glut_motion);
    glutPassiveMotionFunc(glut_motion);
    glutIdleFunc(glut_idle);
}

void glut_keyboard(unsigned char key, int x, int y){
    switch(key){
        case 'q':
        case 'Q':
        case '\033':
            exit(0);
    }
    glutPostRedisplay();
}

void glut_mouse(int button, int state, int x, int y){
    if(button == GLUT_LEFT_BUTTON){
        if(state == GLUT_UP){
            g_isLeftButtonOn = false;
        }else if(state == GLUT_DOWN){
            g_isLeftButtonOn = true;
        }
    }
    
    if(button == GLUT_RIGHT_BUTTON){
        if(state == GLUT_UP){
            g_isRightButtonOn = false;
        }else if(state == GLUT_DOWN){
            g_isRightButtonOn = true;
        }
    }
}

void glut_motion(int x, int y){
    static int px = -1, py = -1;
    if(g_isLeftButtonOn == true){
        if(px >= 0 && py >= 0){
            g_angle1 += (double)-(x - px)/20;
            g_angle2 += (double)(y - py)/20;
        }
        px = x;
        py = y;
    }else if(g_isRightButtonOn == true){
        if(px >= 0 && py >= 0){
            g_distance += (double)(y - py)/20;
        }
        px = x;
        py = y;
    }else{
        px = -1;
        py = -1;
    }
    glutPostRedisplay();
}

void glut_display(){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(30.0, 1.0, 0.1, 100);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(g_distance * cos(g_angle2) * sin(g_angle1),
              g_distance * sin(g_angle2),
              g_distance * cos(g_angle2) * cos(g_angle1),
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    draw_pyramid();
    glFlush();
    glDisable(GL_DEPTH_TEST);
    
    glutSwapBuffers();
}

void glut_idle(){
    cv::Mat frame;
    cap >> frame;
    //cv::imshow("window1", frame);
    if (frame.empty()) {
        printf("window empty!\n");
    }else{
        cv::cvtColor(frame,frame, CV_BGR2RGB);
        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - frame.cols)/2, (TEXTURE_HEIGHT- frame.rows)/2, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
    }
    glutPostRedisplay();
}

void draw_pyramid(){
    GLdouble pointO[] = {0.0, 1.0, 0.0};
    GLdouble pointA[] = {1.5, -1.0, 1.5};
    GLdouble pointB[] = {-1.5, -1.0, 1.5};
    GLdouble pointC[] = {-1.5, -1.0, -1.5};
    GLdouble pointD[] = {1.5, -1.0, -1.5};
    
    glColor3d(1.0, 0.0, 0.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointA);
    glVertex3dv(pointB);
    glEnd();
    
    glColor3d(1.0, 1.0, 0.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointB);
    glVertex3dv(pointC);
    glEnd();
    
    glColor3d(0.0, 1.0, 1.0);
    glBegin(GL_TRIANGLES);
    glVertex3dv(pointO);
    glVertex3dv(pointC);
    glVertex3dv(pointD);
    glEnd();
    
    glEnable(GL_TEXTURE_2D);
    
    glColor3d(1.0, 0.0, 1.0);
    glBegin(GL_TRIANGLES);
    glTexCoord2d(1.0, 1.0);
    glVertex3dv(pointO);
    glTexCoord2d(0.0, 0.0);
    glVertex3dv(pointD);
    glTexCoord2d(1.0, 0.0);
    glVertex3dv(pointA);
    glEnd();
    
    glColor3d(1.0, 1.0, 1.0);
    
    
    
    glBegin(GL_POLYGON);
    glTexCoord2d(1.0, 0.0);
    glVertex3dv(pointA);
    glTexCoord2d(0.0, 0.0);
    glVertex3dv(pointB);
    glTexCoord2d(0.0, 1.0);
    glVertex3dv(pointC);
    glTexCoord2d(1.0, 1.0);
    glVertex3dv(pointD);
    glEnd();
    
    glDisable(GL_TEXTURE_2D);
}
//void set_texture(){
//    const char* inputFileNames[3] = {"/Users/naoto/git/opencv_gl/opencv/fruits.jpg", "/Users/naoto/git/opencv_gl/opencv/hoge3.jpg", "/Users/naoto/git/opencv_gl/opencv/hoge6.jpg"};
//    for(int i=0; i<3; i++){
//        cv::Mat input = cv::imread(inputFileNames[i],1);
//        // BGR -> RGBの変換
//        cv::cvtColor(input, input, CV_BGR2RGB);
//
//        glBindTexture(GL_TEXTURE_2D, g_TextureHandles[i]);
//        glTexSubImage2D(GL_TEXTURE_2D, 0, (TEXTURE_WIDTH - input.cols)/2, (TEXTURE_HEIGHT- input.rows)/2, input.cols, input.rows, GL_RGB, GL_UNSIGNED_BYTE, input.data);
//
//    }
//}

