//include二行と、build_targetをOSx.10.8以下,後はpathにOPENGLとGLUTを入れればOK
#include <stdio.h>
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#define WINDOW_X (500)
#define WINDOW_Y (500)
#define WINDOW_NAME "test1"

void init_GL(int argc, char *argv[]);
void init();
void set_callback_functions();

void glut_display();
void glut_keyboard(unsigned char key, int x, int y);

void draw_square1();
void draw_square2();
void draw_square3();

// グローバル変数
int g_display_mode = 1;

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

void init_GL(int argc, char *argv[]){
    glutInit(&argc,argv);// OpenGLの初期化
    glutInitDisplayMode(GLUT_RGBA);           // ディスプレイモードをRGBAモードに設定　多分最後のAは透過色用
    glutInitWindowSize(WINDOW_X, WINDOW_Y);//ウィンドウサイズを指定
    glutCreateWindow(WINDOW_NAME);// ウィンドウを「生成」。まだ「表示」はされない。
}

void init(){
    glClearColor(0.0, 0.0,0.0, 0.0);// 背景の塗りつぶし色を指定　この場合ただの真っ黒
}

void set_callback_functions(){//この関数内で指定したものを行わないと、MainLoopが終了しない
    glutDisplayFunc(glut_display);// ディスプレイに変化があった時に呼ばれるコールバック関数を登録
    glutKeyboardFunc(glut_keyboard);// キーボードに変化があった時に呼び出されるコールバック関数を登録
}

// キーボードに変化があった時に呼び出されるコールバック関数。
void glut_keyboard(unsigned char key, int x, int y){
    switch(key){
            
        case 'q':
        case 'Q':
        case '\033': // Escキーのこと
            exit(0);
        case '1':
            g_display_mode = 1;
            break;
        case '2':
            g_display_mode = 2;
            break;
        case '3':
            g_display_mode = 3;
            break;
    }
    
    glutPostRedisplay(); // 「ディスプレイのコールバック関数を呼んで」と指示する。描画のためのフラグ設定をしているにすぎない
}

// ディスプレイに変化があった時に呼び出されるコールバック関数。
// 「ディスプレイに変化があった時」は、glutPostRedisplay() で指示する。
void glut_display(){
    glClear(GL_COLOR_BUFFER_BIT); // 今まで画面に描かれていたものを消す
    
    switch(g_display_mode){
        case 1:
            draw_square1();
            break;
        case 2:
            draw_square2();
            break;
        case 3:
            draw_square3();
            break;
    }
    
    glFlush(); // ここで画面に描画をする
}

void draw_square1(){
    glBegin(GL_LINE_LOOP);
    glColor3d(1.0, 0.0, 0.0);
    glVertex2d(-0.9, -0.9);
    glVertex2d(0.9, -0.9);
    glVertex2d(0.9, 0.9);
    glVertex2d(-0.9, 0.9);
    glEnd();
}

void draw_square2(){
    glBegin(GL_POLYGON);
    glColor3d(1.0, 0.0, 0.0);
    glVertex2d(-0.9, -0.9);
    glVertex2d(0.9, -0.9);
    glVertex2d(0.9, 0.9);
    glVertex2d(-0.9, 0.9);
    glEnd();
}

void draw_square3(){
    glBegin(GL_POLYGON);
    glColor3d(1.0, 0.0, 0.0);
    glVertex2d(-0.9, -0.9);
    glColor3d(1.0, 1.0, 0.0);
    glVertex2d(0.9, -0.9);
    glColor3d(0.0, 1.0, 1.0);
    glVertex2d(0.9, 0.9);
    glColor3d(0.0, 0.0, 0.0);
    glVertex2d(-0.9, 0.9);
    glEnd();
}
