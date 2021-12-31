CC = arm-linux-androideabi-g++
ADB = adb

OPENCL_PATH = /home/ubuntu/UOS/MPCLASS/FinalProject/OpenCL_lib_and_include
CFLAG = -I$(OPENCL_PATH)/include -g
LDFLAGS = -l$(OPENCL_PATH)/lib/libGLES_mali.so -lm

TARGET = ProjectGPU
TARGET_SRC = $(TARGET).cpp bmp.cpp MyOpencl.cpp

all: $(TARGET)

$(TARGET): $(TARGET_SRC)
	$(CC) -static $(TARGET_SRC) $(CFLAG) $(LDFLAGS) -fpermissive -o $(TARGET)
	echo
	echo "**** Install:" /data/local/tmp/$(TARGET)"****"
	$(ADB) push $(TARGET) /data/local/tmp
	$(ADB) push letter.bmp /data/local/tmp
	$(ADB) push conv1.txt /data/local/tmp
	$(ADB) push conv2.txt /data/local/tmp
	$(ADB) push linear1.txt /data/local/tmp
	$(ADB) push linear2.txt /data/local/tmp
	$(ADB) push Project.cl /data/local/tmp
	$(ADB) shell chmod 755 /data/local/tmp/$(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)