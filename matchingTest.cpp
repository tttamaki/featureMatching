#include <iostream>
#include <vector>
#include <string>
//#include <fstream>
#include <algorithm>
#include <sstream>

//#include <cstdio>
//#include <cmath>
//#include <cfloat>


#include "option.hxx"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>


using namespace std;
using namespace cv;





template <class T>
inline std::string toStr (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}


float distancePt2f(const cv::Point2f pt1,
                   const cv::Point2f pt2)
{
    float tmp1 = pt1.x - pt2.x;
    float tmp2 = pt1.y - pt2.y;
    return sqrt( tmp1*tmp1 + tmp2*tmp2 );
}

void refineMatches(
    const std::vector<cv::KeyPoint>& query,
    const std::vector<cv::KeyPoint>& train,
    const vector< vector<cv::DMatch> >&  knnMatches,
    vector<DMatch>& matches
)
{
    matches.clear();
    matches.resize(knnMatches.size());
    
    for(int i=0; i<knnMatches.size(); i++){
        if(knnMatches[i].size() < 2) assert("knn is not k=2;");
        
        matches[i] = knnMatches[i][0];
        
        if (knnMatches[i][0].distance / knnMatches[i][1].distance > 0.4 ||
            distancePt2f(query[matches[i].queryIdx].pt,
                         train[matches[i].trainIdx].pt) > 250
        )
            matches[i].trainIdx = -1; // no matche
    }
}



void drawKeypointLines(
    cv::Mat& img,
    const std::vector<cv::KeyPoint>& query, // current frame
    const std::vector<cv::KeyPoint>& train, // last frame
    vector<cv::DMatch>& matches,
    cv::Point2f offset = cv::Point2f(0,0)
)
{
    vector<cv::DMatch>::iterator it;
    
    for ( it=matches.begin() ; it < matches.end(); it++ ){
        if( (*it).trainIdx > 0) {
            int qIdx = (*it).queryIdx;
            int tIdx = (*it).trainIdx;
            
            cv::circle(img, query[qIdx].pt + offset, 3, cv::Scalar(0,200,255) );
            //cv::circle(img, train[tIdx].pt + offset, 3, cv::Scalar(0,200,255) );
            cv::line(  img, query[qIdx].pt + offset, train[tIdx].pt + offset, cv::Scalar(0,200,255));
        }
    }
}






int main ( int argc, char **argv )
{
    
    options Option = parseOptions(argc, argv);
    
    
    cv::VideoCapture cap;
    
    if (Option.cameraID >= 0)
        cap = VideoCapture(Option.cameraID); // from camera
    else
        cap = cv::VideoCapture(Option.filename); // from movie
    
    
    
    
    
    std::cout << "Press 'q' key to exit." << std::endl;
    
    std::string detectorNamePrefix = ""; // default
    std::string detectorName = "SURF"; // default
    std::string descriptorName = "SURF"; //default
    std::string matcherName = "FlannBased"; //default
    
    
    cv::Mat img, frame;
    cv::Mat gray_img;
    std::vector<cv::KeyPoint> keypoints, keypointsLast;
    cv::Mat features, featuresLast;
    vector<cv::DMatch> matches, matchesLast;
    
    cv::Mat roi, roiLast; // ROI: Region-Of-Interest
    
    cv::Size patchSize(250,250);
    cv::Point patchCenter(320,240);
    cv::Point2f ulROIcorner = Point2f(patchCenter.x-patchSize.width/2, patchCenter.y-patchSize.height/2);
    cv::Point2f brROIcorner = Point2f(patchCenter.x+patchSize.width/2, patchCenter.y+patchSize.height/2);
    
    cv::namedWindow("detector/descriptor/matcher", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
        

    
    long frameCounter = 0;
    
    if(Option.startframe != 1){
        cout << "skip to " << Option.startframe << endl;
        while(frameCounter++ < Option.startframe){
            cap.grab();
        }
    }
    
    long skipFrames = 1;
    frameCounter = 0;  // rewind
    while ( 1 )
    {
        for (int i=0; i<skipFrames; i++) cap >> frame;
        if ( frame.empty() ) break;
        
        
        cv::cvtColor(frame, gray_img, CV_BGR2GRAY);
        cv::normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);



        cv::getRectSubPix ( gray_img, patchSize, patchCenter, roi );
        cv::rectangle(frame,
                      ulROIcorner,
                      brROIcorner,
                      cv::Scalar(100,200,255) );
        // cv::imshow("ROI", roiLast);



        //
        // keypoint detection
        //
        try{
            Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( detectorNamePrefix + detectorName );
            //detector->detect(gray_img, keypoints); // full frame
            detector->detect(roi, keypoints); // only in the ROI
        }catch(...){}
        
        //
        // feature description
        //
        try{
            Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create( descriptorName );
            descriptor->compute(gray_img, keypoints, features);
        }catch(...){}      

        //
        // matching strategy
        //
        try{
            Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create( matcherName );

            if (frameCounter >= 1)
            {
                vector< vector<cv::DMatch> > knnMatches;
                matcher->knnMatch(features, featuresLast, knnMatches, 2);
                refineMatches(keypoints, keypointsLast, knnMatches, matches);
            }

        }catch(...){}


        
        
        
        img = frame.clone();
        try{
            drawKeypointLines(img, keypoints, keypointsLast, matches, ulROIcorner);
        }
        catch(...){
        }
        
        cv::putText (img, 
                     " detector: " + detectorNamePrefix + detectorName + 
                     " / descriptor: " + descriptorName + 
                     " / matching: " + matcherName + 
                     "  (skip " + toStr<long>(skipFrames) + " frames)",
                     Point ( 10, 50 ), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB ( 0, 0, 0 ) );

        
        
        
        cv::imshow("detector/descriptor/matcher", img);
        
        
        //
        // copy current frame infomation 
        //
        keypointsLast.resize(keypoints.size());
        std::copy (keypoints.begin(), keypoints.end(), keypointsLast.begin() );
        
        featuresLast = features.clone();
        
        matchesLast.resize(matches.size());
        std::copy (matches.begin(), matches.end(), matchesLast.begin() );
        
        roiLast = roi.clone();
        
        
        
        
        
        frameCounter++;
        
        
        bool isBreak = false;
        int key;
        if( (key = cv::waitKey(30)) >= 0) {
            //             std::cout << key << std::endl;
            switch (key)
            {
                case 'q' : isBreak = true; break;
                case 'p' : skipFrames++; break;
                case '@' : skipFrames = (skipFrames>1) ? skipFrames-1 : 1; break;
                case 'l' : for (int i=0; i<200; i++) cap >> frame; break;
                case 'k' : skipFrames = (skipFrames < 10) ? 200 : 1; break;

                
                // Note
                // cv::Feature2D = cv::FeatureDetector + cv::DescriptorExtractor
                //

                // detector selection
                case '1' : detectorName = "FAST" ; break;
                case '!' : detectorName = "FASTX" ; break;
                case '2' : detectorName = "STAR" ; break;
                case '3' : detectorName = descriptorName = "SIFT" ; break;
                case '4' : detectorName = descriptorName = "SURF" ; break;
                case '5' : detectorName = descriptorName = "ORB"  ; break;
                case '6' : detectorName = "MSER" ; break;
                case '7' : detectorName = "GFTT" ; break;
                case '8' : detectorName = "HARRIS" ; break;
                case '9' : detectorName = descriptorName = "BRISK" ; break;
                case '0' : detectorName = "SimpleBlob" ; break;
                
                // adapter selection
                case 'w' : detectorNamePrefix = "Grid" ; break;
                case 'e' : detectorNamePrefix = "Pyramid" ; break;
                //case 'r' : detectorNamePrefix = "Dynamic" ; break;
                case 't' : detectorNamePrefix = "" ; break; // default adaptor
                
                // descriptor selection
                case 'a' : descriptorName = "SIFT"; break;
                case 's' : descriptorName = "SURF"; break;
                case 'd' : descriptorName = "ORB"; break;
                case 'f' : descriptorName = "BRISK"; break;
                case 'g' : descriptorName = "FREAK"; break;
                case 'h' : descriptorName = "BRIEF"; break;
                case 'j' : descriptorName = "Opponent"; break;

                
                
                // matcher selection
                case 'z': matcherName = "FlannBased" ; break;
                case 'x': matcherName = "BruteForce"  ; break;
                case 'c': matcherName = "BruteForce-SL2"  ; break;
                case 'v': matcherName = "BruteForce-L1"  ; break;
                case 'b': matcherName = "BruteForce-Hamming" ; break;
                case 'n': matcherName = "BruteForce-HammingLUT" ; break;
                
                
                default : break;
            }
            
            if (isBreak) break;
            
            frameCounter = 0;
            
        }
        
        
    }
    
    return 0;
}







