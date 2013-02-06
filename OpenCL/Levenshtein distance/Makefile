all:main.o
	g++ -o Levenshtein_distance main.o -lOpenCL

main.o: main.cpp
	g++ -c main.cpp -I/usr/local/cuda/include
