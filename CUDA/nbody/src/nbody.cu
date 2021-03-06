//! I used some of Nvidia code to display bodies in OpenGL.
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */
 
 
#include "nbody_kernel.cu"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_runtime.h>


//! vbo variables
GLuint vbo;

//! GPU arrays
float4 *Positions;
float *Mass;
float4 *VelocityVector;
float *Borders;
float *boundingTemporary;

int schema = 0;

//! mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;



// declaration
void run( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVBO( GLuint* vbo);
void deleteVBO( GLuint* vbo);

//! rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

//! Cuda functionality
void runCuda( GLuint vbo);
void initCuda( GLuint vbo);
void checkResultCuda( int argc, char** argv, const GLuint& vbo);

int counter = 0;

int main( int argc, char** argv) {
    CUT_DEVICE_INIT(argc, argv);

    run(argc, argv);

    CUT_EXIT(argc, argv);
}

void run( int argc, char** argv) {

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "N-body");

    // initialize GL
    if ( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // create VBO
    createVBO( &vbo);

    initCuda( vbo);
    runCuda( vbo);

    // check result of Cuda step
    checkResultCuda( argc, argv, vbo);

    // start rendering mainloop
    glutMainLoop();
}

//! Run the Cuda part of the computation
void runCuda( GLuint vbo){
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, vbo));


    // execute the kernels
    int blocks = bodies/threads;
    
    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer(   timer));
    improvedGravity<<<blocks, threads>>>(Positions, VelocityVector, Mass);
        
    
    cudaThreadSynchronize();
    if(counter % 50 == 0) {
      printf( "single iteration GPU time: %2.6f (ms)\n", cutGetTimerValue( timer));
    }
    cutilCheckError( cutDeleteTimer( timer));
    
    simpleMove<<<blocks, threads>>>(dptr, Positions, VelocityVector);
    
    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
}

void initCuda( GLuint vbo) {
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, vbo));


    cudaMalloc(&Positions, (bodies * sizeof(*Positions)));
    cudaMalloc(&Mass, (bodies * sizeof(*Mass)));
    cudaMalloc(&VelocityVector, (bodies * sizeof(*VelocityVector)));
    cudaMalloc(&Borders, (6 * sizeof(*Borders)));
    cudaMalloc(&boundingTemporary, (6 * bodies * sizeof(*boundingTemporary)));
    srand(time(0));
    int blocks = bodies/threads;
    unsigned int seed = rand();
    if(schema % 5 == 0) explosion<<<blocks, threads>>>(dptr, Positions, seed, VelocityVector, Mass);
    if(schema % 5 == 1) heavyMiddle<<<blocks, threads>>>(dptr, Positions, seed, VelocityVector, Mass);
    if(schema % 5 == 2) randomStatic<<<blocks, threads>>>(dptr, Positions, seed, VelocityVector, Mass);
    if(schema % 5 == 3) randomMoving<<<blocks, threads>>>(dptr, Positions, seed, VelocityVector, Mass);
    if(schema % 5 == 4) explosion2<<<blocks, threads>>>(dptr, Positions, seed, VelocityVector, Mass);

    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
}

//! Initialize GL
CUTBoolean initGL(){
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 "
                           "GL_ARB_pixel_buffer_object"
                         )) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 100.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

//! Create VBO
void createVBO(GLuint* vbo) {
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = bodies * 4 * sizeof( float);
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

    CUT_CHECK_ERROR_GL();
}

//! Delete VBO
void deleteVBO( GLuint* vbo) {
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));

    *vbo = 0;
}

//! Display callback
void display(){
    counter += 1;

    // run CUDA kernel to generate vertex positions
    runCuda(vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.10, 0.60, 0.70);
    glPointSize(1);
    glDrawArrays(GL_POINTS, 0, bodies);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();


}

//! Keyboard events handler
void keyboard( unsigned char key, int /*x*/, int /*y*/){
    switch ( key) {
    case( 27) :
        deleteVBO( &vbo);
        exit( 0);
    case(' ') :
        schema++;
	initGL();
        initCuda (vbo);
    case('r') :
        initCuda (vbo);
    }
}

//! Mouse event handlers
void mouse(int button, int state, int x, int y){
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y){
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

//! Check if the result is correct or write data to file for external
//! regression testing
void checkResultCuda( int argc, char** argv, const GLuint& vbo){
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));

    // map buffer object
    glBindBuffer( GL_ARRAY_BUFFER_ARB, vbo );
    float* data = (float*) glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY);

    // check result
    if ( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) {
        // write file for regression test
        CUT_SAFE_CALL( cutWriteFilef( "./data/regression.dat",
                                      data, bodies * 3, 0.0));
    }

    // unmap GL buffer object
    if ( ! glUnmapBuffer( GL_ARRAY_BUFFER)) {
        fprintf( stderr, "Unmap buffer failed.\n");
        fflush( stderr);
    }

    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));

    CUT_CHECK_ERROR_GL();
}
