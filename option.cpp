
#include "option.hxx"
using namespace std;


options parseOptions( int argc, char **argv ) {
    
    namespace po = boost::program_options;
    
    po::options_description desc("Options");
    desc.add_options()
    ("help", "This help message.")
    ("input", po::value<string>(), "Input filename.")
    ("startframe", po::value<int>(), "start frame number.")
    ("cameraID", po::value<int>(), "camera ID (default: -1, no camera is used)")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    options Opt;
    
    if (vm.count("help")) {
        cout << desc << endl;
        exit(0);
    }
    

    
    Opt.cameraID = -1;
    if(vm.count("cameraID")) {
        Opt.cameraID = vm["cameraID"].as<int>();
        cout << "cameraID: " << Opt.cameraID << endl;
    }

    Opt.startframe = 1;
    if(vm.count("startframe")) {
        Opt.startframe = vm["startframe"].as<int>();
        cout << "start frame number: " << Opt.startframe << endl;
    }
    
    if (vm.count("input")) {
        Opt.filename = vm["input"].as<string>();
        cout << "    input filename: " << Opt.filename << endl;
    } else if (Opt.cameraID == -1) {
        cout << "no input is specified." << endl
        << desc << endl;
        exit(1);
    }
    

        
    
    return Opt;
}
