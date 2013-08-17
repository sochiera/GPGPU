all: rot13

rot13: main.o
	gcc main.o -o rot13 -lOpenCL

main.o: main.c
	gcc -c -std=c99 -L/opt/AMDAPP/lib/x86_64 -I/opt/AMDAPP/include main.c