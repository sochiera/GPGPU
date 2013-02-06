__kernel void myKernel(__global int* wy, int tabSize)
{
  int id=get_global_id(0);
  if(id>tabSize) return;

  wy[id]=id;
}


__kernel void StrLenGpuKernel(__global char* _Dict, __global char* Lengths){
  int dictSize = 796032;
  int x = get_global_id(0);
  __global char *w = _Dict + x;
  int i = 0;
  int j = 0;
  while(w[j] != '\0')
  {
    i++;
    j += dictSize;
  }
  Lengths[x] = i;
}



__kernel void LevenshteinDistanceGpuKernel(__global char* _Dict, __constant char* Word, __global char* Lengths, __global char* Results, __global char* Temp1, __global char* Temp2)
{  
  
  int dictSize = 796032;
  int x = get_global_id(0);

  __global char *t1 = Temp1 + x;
  __global char *t2 = Temp2 + x;
  
  __global char * const b = _Dict + x;
  const int aN = Word[16];
  const int bN = Lengths[x];
  
  __global char *d1 = t1;
  __global char *d2 = t2;
  int j = 0;  
  while(j<=bN) {
    d1[dictSize*j]=j;
    j++;
  }
  int i=1;
  while(i<=aN) {
     d2[0] = i;
      j = 1;
      while(j<=bN) {
	char ra = d1[j * dictSize] + 1;                       
	char rb = d2[(j-1) * dictSize] + 1;                    
	char rc = d1[(j-1) * dictSize];
	if((Word[i-1] != b[dictSize*(j-1)])){
	  rc += 1;
	}
	
	
	if(ra < rb) rb = ra;
	if(rc < rb) rb = rc;
	d2[j * dictSize] = rb;
	j++;
      }
      d1 = (d1==t1)? t2:t1;
      d2 = (d2==t2)? t1:t2;
      i++;
  }
  
  Results[x] = d1[bN * dictSize];
}

