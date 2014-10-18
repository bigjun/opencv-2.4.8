CC = g++
CFLAGS = -W
LDFLAGS = -lopencv_photo \
		  -lopencv_core \
		  -lopencv_highgui \
		  -lopencv_imgproc

TARGET = exampleImg
TARGET += exampleSmooth
TARGET += exampleCanny

all:$(TARGET)

exampleImg:
	$(CC) exampleImg.cpp $(CFLAGS) $(LDFLAGS) -o $@

exampleSmooth:
	$(CC) exampleSmooth.cpp $(CFLAGS) $(LDFLAGS) -o $@

exampleCanny:
	$(CC) exampleCanny.cpp $(CFLAGS) $(LDFLAGS) -o  $@

clean:
	rm $(TARGET)

