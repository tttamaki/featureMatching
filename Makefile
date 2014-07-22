all: matchingTest

clean:
	rm -f *.o matchingTest


#OPTFLAG = -O0 -g -ggdb
OPTFLAG = -O3 -g -ggdb

CXXFLAGS:= `pkg-config --cflags opencv` `pkg-config --cflags eigen3` -I/opt/local/include $(OPTFLAG) -Wall
#LDFLAGS:= -lboost_program_options -lboost_system  -L/usr/lib64/  `pkg-config --libs opencv` 
LDFLAGS:= -L/opt/local/lib -lboost_program_options-mt -lboost_system-mt `pkg-config --libs opencv`


matchingTest: matchingTest.o option.o
	$(CXX) -o matchingTest \
	matchingTest.o option.o \
	$(LDFLAGS)
