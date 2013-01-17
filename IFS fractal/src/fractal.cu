// nvcc -lglut -lGLEW -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc -L/opt/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_i386 fraktal.cu
// Compile ^^^
// 
// =======================================================================
//    Class: Procesory graficzne w obliczeniach równoległych (CUDA)
//    Task: IFS (list 5)
//    
//    Jan Sochiera, 241745
//    
//    "whitenoise.cu" code reused
//  ----------------------------------------------------------------------  
//    
//    Usage: 
//    	i 	      - zoom in
//    	o 	      - clear and generate default view
//    	a-s and z-x   - modify matrix 
//    	q 	      - quit
//    	any other     - concentrate fractal
//	
//========================================================================
#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cutil.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>

typedef unsigned int  uint;

#define sieveSize 32 
#define red 255
#define pointsPerThread 48
#define imageWidth 1366
#define imageHeight 742
#define imageH imageHeight
#define imageW imageWidth

float realWidth = 3.31;
float realHeight = 1.8;
float s = 0.97;

__device__ __constant__ float Matrix[12];

//=========================================================================
// Pseudo Random: TausStep, LCGStep, Hybrid based on:
// GPU Gems 3:
// Lee Howes, David Thomas (Imperial College London)
// Chapter 37. Efficient Random Number Generation and Application Using CUDA
//=========================================================================
// Cheap pseudo random numbers:
//  
// S1, S2, S3, M - constants,  z - state
__device__ uint TausStep(uint &z, int S1, int S2, int S3, uint M)  {
    uint b=(((z << S1) ^ z) >> S2);
    return z = (((z & M) << S3) ^ b);
}

// A, C - constants
__device__ uint LCGStep(uint &z, uint A, uint C) {
    return z=(A*z+C);
}

// Mixed :
__device__ float HybridTaus(uint &z1, uint &z2, uint &z3, uint &z4) {
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    return 2.3283064365387e-10 * (              // Periods
               TausStep(z1, 13, 19, 12, 4294967294UL) ^   // p1=2^31-1
               TausStep(z2,  2, 25,  4, 4294967288UL) ^   // p2=2^30-1
               TausStep(z3,  3, 11, 17, 4294967280UL) ^   // p3=2^28-1
               LCGStep( z4,    1664525, 1013904223UL)     // p4=2^32
           );
}

// Int Mixed and modified: cheaper
__device__ uint HybridTausInt(uint &z1, uint &z2, uint &z3, uint &z4) {
    // Combined period is lcm(p1,p2,p3,p4)~ 2^121
    return (              // Periods
               TausStep(z1, 13, 19, 12, 4294967294UL) ^   // p1=2^31-1
               //  TausStep(z2,  2, 25,  4, 4294967288UL) ^   // p2=2^30-1
               //  TausStep(z3,  3, 11, 17, 4294967280UL) ^   // p3=2^28-1
               LCGStep( z4,    1664525, 1013904223UL)     // p4=2^32
           );
}

// Testing func:   cheap one int state
__device__ uint funct(uint id) {
    //return LCGStep( id,    1664525, 1013904223UL) ;    // p4=2^32
    return HybridTausInt(id,id,id,id);
    //return id = (1664525*id+1013904223UL) % (65536*256);
    //return id = (xx%256) + 256*(y%256) + 65536*( (256-(xx%256)-(y%256))%256 ) ;
}
//=========================================================================

//=========================================================================
//initialization kernel:
__global__ void initim1(uint * output) {
    uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint id = __umul24(y, imageWidth) + x;  // Good for < 16MPix

    if ( x < imageWidth && y < imageHeight ) {
	output[id] = 0;
    }
}

//temp to output DeviceToDevice fast(?) copy
__global__ void outputCpy(uint * output, uint * temp) {
    uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint id = __umul24(y, imageWidth) + x;  // Good for < 16MPix
    if ( x < imageWidth && y < imageHeight ) {
	output[id] = temp[id];
    }
}

#define scale(x, y, fx, fy)\
fx = float(x - (imageW/2)) * realW/(1.0*float(imageW));\
fy = float(y - (imageH/2)) * realH/(1.0*float(imageH));

#define rescale(x, y, fx, fy)\
x = (fx) * float(imageW)/realW + imageW/2;\
y = (fy) * float(imageH)/realH + imageH/2;

//computes and concentrates fractal
__global__ void fractalCompute(uint * output, float realW, float realH, uint seed) {

  int ix = (blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);
  uint xSeed  = funct(ix*seed);
  uint ySeed  = funct(iy*seed);
  
  float fx, fy;
  scale(ix, iy, fx, fy);
  
  for(int i = 0; i < pointsPerThread; i++){
    float x = fx;
    float y = fy;
    if(funct(uint(i * seed* 12431 + 327911*xSeed + 98097*ySeed)) % 22519117 > 11500000){
      fx = Matrix[0]*x + Matrix[1]*y + Matrix[2];
      fy = Matrix[3]*x + Matrix[4]*y + Matrix[5];
    }
    else{
      fx = Matrix[6]*x + Matrix[7]*y + Matrix[8];
      fy = Matrix[9]*x + Matrix[10]*y + Matrix[11];
    }
    rescale(ix, iy, fx, fy);
    
    if (i >= sieveSize && abs(fx*2.0) < realW && abs(fy*2.0 ) < realH && ix < imageW && iy < imageH){
      output[iy*imageW + ix] = red;
    }
  }
}



//simply interpolates output and save shrinked view in temp
//calibrated points blackens over time
__global__ void imageScale(uint * output, float realW, float realH, float s, uint *temp) {
  int ix = (blockIdx.x * blockDim.x + threadIdx.x);
  int iy = (blockIdx.y * blockDim.y + threadIdx.y);

  float fx, fy;
  scale(ix, iy, fx, fy);

  int color = 0;
  if (abs(fx*2.0) < realW && abs(fy*2.0 ) < realH && ix < imageW && iy < imageH) color = output[iy*imageW + ix];
  if(color > 5){

    realW *= s;
    realH *= s;
    rescale(ix, iy, fx, fy);
    if (abs(fx*2.0) < realW && abs(fy*2.0 ) < realH && ix < imageW && iy < imageH){
      temp[iy*imageW + ix] = (color - 25);
    }
  }
}


float A[12] = {-0.4, 0, -1, 0, -0.4, 0.1, 0.76, -0.4, 0, 0.4, 0.76, 0};

GLuint   pbo = 0;      // OpenGL PBO id.
uint    *output;   // CUDA device pointer to PBO data
uint 	*tempOutput;

dim3 blockSize(16,16); // threads
dim3 gridSize;         // set up in initPixelBuffer

int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer() {
    if (pbo) {      // delete old buffer
        cudaGLUnregisterBufferObject(pbo);
        glDeleteBuffersARB(1, &pbo);
    }
    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, imageWidth * imageHeight * sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    cudaGLRegisterBufferObject(pbo);

    // calculate new grid size
    gridSize = dim3(iDivUp(imageWidth, blockSize.x), iDivUp(imageHeight, blockSize.y));

    // from display:
    cudaGLMapBufferObject((void**)&output, pbo  );
    initim1<<<gridSize, blockSize>>>(output);
    CUT_CHECK_ERROR("Kernel error");
    cudaGLUnmapBufferObject(pbo);
    cudaMalloc(&tempOutput, imageW*imageH*sizeof(uint));
    initim1<<<gridSize, blockSize>>>(tempOutput);
    CUT_CHECK_ERROR("Kernel error");
}

static int cnt=0; // generation(display calls) count

void display() {
    printf("%4d\n", cnt % 10000);
    cnt++;

    cudaGLMapBufferObject((void**)&output, pbo);
   
    int random = rand();
    
    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer(   timer));
    
    fractalCompute<<<gridSize, blockSize>>>(output, realWidth, realHeight, uint(random));

    cudaThreadSynchronize();
    printf( "GPU time: %2.6f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));
    
    CUT_CHECK_ERROR("Kernel error");
    cudaGLUnmapBufferObject(pbo );

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawPixels(imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void reshape(int x, int y) {

    initPixelBuffer();
    glViewport(0, 0, x, y);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}


void keyboard(unsigned char k, int , int ) {
    if (k == 27 || k == 'q' || k == 'Q') exit(1);
    
    if (k == 'i' || k == 'I'){
//      initim1<<<gridSize, blockSize>>>(tempOutput);
//      imageScale<<<gridSize, blockSize>>>(output, realWidth, realHeight, s, tempOutput);  
      realWidth *= s;
      realHeight *= s; 
      initim1<<<gridSize, blockSize>>>(output);  
//      outputCpy<<<gridSize, blockSize>>>(output, tempOutput);  
//      cudaMemcpy(output, tempOutput, imageH * imageW * sizeof(uint), cudaMemcpyDeviceToDevice);
    }
    
    if (k == 'o' || k == 'O'){
      initim1<<<gridSize, blockSize>>>(output);
      realWidth = 3.2;
      realHeight = 1.8;
      A[3] = 0;
      A[4] = -0.4;
      cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A)); 
    }
    
    if (k == 'z'){
      initim1<<<gridSize, blockSize>>>(output);
      A[3] += 0.01;
      cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A)); 

    }
    
    
    if (k == 'x'){
      initim1<<<gridSize, blockSize>>>(output);
      A[3] -= 0.01;
      cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A)); 
    }
    
    if (k == 'a'){
      initim1<<<gridSize, blockSize>>>(output);
      A[4] += 0.01;
      cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A)); 
    }
    
    
    if (k == 's'){
      initim1<<<gridSize, blockSize>>>(output);
      A[4] -= 0.01;
      cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A));
    }
    
    display();
}

void cleanup() {
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffersARB(1, &pbo);
}

int main( int argc, char** argv) {
  
    srand(1337);
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    int windowWidth = imageW;
    int windowHeight = imageH;
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Fractal");
    
    cudaMemcpyToSymbol("Matrix", A, 12*sizeof(*A)); 
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "OpenGL requirements not fulfilled !!!\n");
        exit(-1);
    }
    initPixelBuffer();

    atexit(cleanup);
    glutMainLoop();
    return 0;
}




















