#include "detector.hpp"

cv::Ptr<Detector> createDetectorHough();
cv::Ptr<Detector> createDetectorThreshold();

//Other detectors

cv::Ptr<Detector> detectorCreation(const std::string &impl_name)
{
    if (impl_name == "Hough")
        return createDetectorHough();
    else if (impl_name == "Threshold")
        return createDetectorThreshold();
    return 0;
}
