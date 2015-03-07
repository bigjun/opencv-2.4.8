CC = gcc
CPP = g++
CFLAGS = -Werror
LDFLAGS = -lopencv_core \
		  -lopencv_highgui \
		  -lopencv_imgproc \
		  -lopencv_contrib \
		  -lopencv_objdetect

TARGET += FaceDetector
TARGET += FaceRecEigenfaces
TARGET += FaceRecFisherfaces
TARGET += FaceRecLbph
TARGET += FaceRecDemo

all:$(TARGET)

FaceDetector:FaceDetector.cpp
	$(CPP) FaceDetector.cpp $(CFLAGS) $(LDFLAGS) -o $@	

FaceRecDemo:FaceRecDemo.cpp
	$(CPP) FaceRecDemo.cpp $(CFLAGS) $(LDFLAGS) -o $@

FaceRecEigenfaces:FaceRecEigenfaces.cpp
	$(CPP) FaceRecEigenfaces.cpp $(CFLAGS) $(LDFLAGS) -o $@

FaceRecFisherfaces:FaceRecFisherfaces.cpp
	$(CPP) FaceRecFisherfaces.cpp $(CFLAGS) $(LDFLAGS) -o $@

FaceRecLbph:FaceRecLbph.cpp
	$(CPP) FaceRecLbph.cpp $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm $(TARGET)
