#ifndef _OPTION_H_
#define _OPTION_H_

#include <iostream>

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"


struct options {
    std::string filename;
    
    int startframe;
    int endframe;
    
    int cameraID;
};

options parseOptions(int argc, char* argv[]);

#endif
