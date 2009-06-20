all: facedetect

facedetect: facedetect.o
	gcc -O2 -o facedetect facedetect.o -lcxcore -lcv -lhighgui -lcvaux -lml

facedetect.o: facedetect.c
	gcc -O2 -c facedetect.c -I/usr/include/opencv

clean:
	rm -f facedetect facedetect.o
