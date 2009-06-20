all: facedetect

facedetect: facedetect.o
	gcc -o facedetect facedetect.o -lcxcore -lcv -lhighgui -lcvaux -lml

facedetect.o: facedetect.c
	gcc -c facedetect.c -I/usr/include/opencv

clean:
	rm -f facedetect facedetect.o
