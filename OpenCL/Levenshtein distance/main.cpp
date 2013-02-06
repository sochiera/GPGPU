//usage: Levenshtein_distance word1 word2 ...

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <math.h>
#include <fstream>
#include <cstring>

using namespace std;

cl_int 			err;
size_t			localSize, globalSize;
cl_platform_id 		platformID;
cl_device_id   		deviceID;
cl_context     		context;
cl_command_queue	queue;

char *dict;
cl_mem			Dict;

char *lengths;
char *results;
cl_mem			Lengths;
cl_mem			Results;
cl_mem			Temp1;
cl_mem			Temp2;
cl_mem			Word;

cl_program		programOpenCL;
cl_kernel		kernel;

char *sourceProgramOpenCL;

const char *dictionaryName = (const char *)("slowa.txt");

const int exactDictSize = 796031;       //długosc naszego slownika
const int wordSize = 16;                //maksymalna dlugosc slowa
const int Threads = 128;     

const int Blocks = exactDictSize/Threads + 1;  
const int dictSize = Blocks*Threads;


void LevenshteinDistanceGpu(char* _word){ 
  
  localSize = Threads;
  globalSize = dictSize;
  
  char WordLength = strlen(_word); 
  
  Word = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * (wordSize+1), NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Word\""<<endl;
    exit(-1);
  }
  char word[17];
  for(int i = 0 ; i < WordLength; i++) word[i] = _word[i];
  word[16] = WordLength;
  
  err=clEnqueueWriteBuffer(queue, Word, CL_TRUE, 0, sizeof(char) * (wordSize+1), word, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy kopiowaniu słowa na kartę graficzną..."<<endl;
    exit(-1);
  } 
  

  kernel=clCreateKernel(programOpenCL, "LevenshteinDistanceGpuKernel", &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu kernela LevenshteinDistanceGpuKernel"<<endl;
    exit(-1);
  }  
  
  err=clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Dict);
  err|=clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Word);
  err|=clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&Lengths);
  err|=clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Results);
  err|=clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&Temp1);
  err|=clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&Temp2);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy ustawianiu parametrów kernela..."<<endl;
    exit(-1);
  }  


  err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy uruchamianiu kernela..."<<endl;
    exit(-1);
  }  

 // LevenshteinDistanceGpuKernel<<<Blocks, Threads>>>(Dict, WordLength, Lengths, Results, Temp1, Temp2);

  
  err=clEnqueueReadBuffer(queue, Results, CL_TRUE, 0, sizeof(cl_char) * globalSize, results, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy kopiowaniu wyników z karty graficznej..."<<endl;
    exit(-1);
  } 

  char odl = 100;

  int wynikPtr;
  for(int i = 0; i < dictSize; i++){
    if(results[i] < odl) {
      odl = results[i];
      wynikPtr = i;
    }
  }

  char wynik[16];
  int ii;
  for(ii = 0; dict[wynikPtr + dictSize*ii] != '\0'; ii++){
     wynik[ii] = dict[wynikPtr + dictSize*ii];
  }
  wynik[ii] = '\0';
  
  cout << "odpowiedz to: " << wynik<< ", odleglosc to " << int(odl) << "\n";
  
}

void computeWordsLengths(){
 
  localSize = Threads;
  globalSize = dictSize;
  
  
  kernel=clCreateKernel(programOpenCL, "StrLenGpuKernel", &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu kernela StrLenGpuKernel"<<endl;
    exit(-1);
  }  
  
  err=clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&Dict);
  err|=clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&Lengths);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy ustawianiu parametrów kernela..."<<endl;
    exit(-1);
  }  



  err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy uruchamianiu kernela..."<<endl;
    exit(-1);
  }  
  
  err=clEnqueueReadBuffer(queue, Lengths, CL_TRUE, 0, sizeof(cl_char) * globalSize, lengths, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy kopiowaniu długości z karty graficznej..."<<endl;
    exit(-1);
  } 

  
}


void initGpuArrays(){
  Dict = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_char) * dictSize * wordSize, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Dict\""<<endl;
    exit(-1);
  }
  
  Lengths = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * dictSize, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Lengths\""<<endl;
    exit(-1);
  }
  Temp1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * dictSize * wordSize, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Temp1\""<<endl;
    exit(-1);
  }
  Temp2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * dictSize * wordSize, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Temp2\""<<endl;
    exit(-1);
  }
  Results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_char) * dictSize, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu bufora \"Results\""<<endl;
    exit(-1);
  }
  err=clEnqueueWriteBuffer(queue, Dict, CL_TRUE, 0, sizeof(cl_char) * dictSize * wordSize, dict, 0, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy kopiowaniu słownika na kartę graficzną..."<<endl;
    exit(-1);
  } 
  
}

void initKernel(){
    err = clGetPlatformIDs(1, &platformID, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy pobieraniu ID platformy..."<<endl;
    exit(-1);
  }
  
  err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy pobieraniu ID urządzenia..."<<endl;
    exit(-1);
  }
  
  context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu kontekstu..."<<endl;
    exit(-1);
  }
  
  queue = clCreateCommandQueue(context, deviceID, 0, &err);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy tworzeniu kolejki komend..."<< endl;
    exit(-1);
  }
  
    // wczytanie pliku cl z dysku
  string line;
  ifstream fileCL ("kernel.cl");
  string sourceProgramOpenCLString="";
  if (fileCL.is_open())
  {
    while ( fileCL.good() )
    {
      getline (fileCL,line);
      sourceProgramOpenCLString+=line;
    }
    fileCL.close();
  }
  sourceProgramOpenCL=new char[sourceProgramOpenCLString.size()];
  sprintf(sourceProgramOpenCL, "%s", sourceProgramOpenCLString.c_str());
  programOpenCL=clCreateProgramWithSource(context, 1, (const char **)&sourceProgramOpenCL, NULL, &err);
  
  
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy przygotowywaniu programu OpenCL do kompilacji..."<<endl;
    exit(-1);
  }
  delete[] sourceProgramOpenCL;
    
  err=clBuildProgram(programOpenCL, 0, NULL, NULL, NULL, NULL);
  if(err!=CL_SUCCESS){
    cout<<"Błąd przy kompilowaniu programu OpenCL..."<<endl;
    cout << err << endl;
    
    exit(-1);
  }  
}

void loadDictionary(const char *name){
  FILE *fp;
  
  for(int i = 0; i < wordSize*dictSize; i++){
    dict[i] = '\0';
  }
  
  if ((fp=fopen(name, "r"))==NULL) {
     cout << "Nie mogę otworzyć pliku!\n";
     exit(1);
  }
  
  char s[wordSize];
  for(int i = 0; i < exactDictSize; i++){
    fscanf(fp, "%s", s);

    int j;
    for(j = 0; s[j] != '\0'; j++){
      dict[j*dictSize + i] = s[j];
    } 
  }

  fclose (fp);
}

void initializeCpuArrays(){
  lengths = (char*)(malloc(dictSize * sizeof(char)));
  results = (char*)(malloc(dictSize * sizeof(char)));
  dict = (char*)(malloc(wordSize * dictSize * sizeof(char)));
}
  
int main(int argc, char** argv){
  initializeCpuArrays();
  loadDictionary(dictionaryName);
  initKernel();
  initGpuArrays();
  
  clock_t start=clock();
  computeWordsLengths();
  for(int i = 1; i < argc; i++){
    LevenshteinDistanceGpu(argv[i]);
  }
  
  clock_t stop=clock();
  cout << "Czas wykonania: " << double((stop-start)*1000.0/(double(CLOCKS_PER_SEC))) << " ms"<< endl;

  return 0;
}
