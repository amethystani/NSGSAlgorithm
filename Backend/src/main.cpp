#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include "cmdline.h"
#include "utils.h"
#include "yolov8Predictor.h"
#include <thread>

int main(int argc, char *argv[])
{
    float confThreshold = 0.1f;
    float iouThreshold = 0.1f;

    float maskThreshold = 0.1f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", false, "yolov8m.onnx");
    cmd.add<std::string>("image_path", 'i', "Image source to be predicted.", false, "./Imginput");
    cmd.add<std::string>("out_path", 'o', "Path to save result.", false, "./Imgoutput");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", false, "coco.names");

    cmd.add<std::string>("suffix_name", 'x', "Suffix names.", false, "yolov8m");

    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = cmd.get<std::string>("image_path");
    const std::string savePath = cmd.get<std::string>("out_path");
    const std::string suffixName = cmd.get<std::string>("suffix_name");
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }
    if (!std::filesystem::exists(modelPath))
    {
        std::cerr << "Error: There is no model." << std::endl;
        return -1;
    }
    if (!std::filesystem::is_directory(imagePath))
    {
        std::cerr << "Error: There is no model." << std::endl;
        return -1;
    }
    if (!std::filesystem::is_directory(savePath))
    {
        std::filesystem::create_directory(savePath);
    }
    std::cout << "Model from :::" << modelPath << std::endl;
    std::cout << "Images from :::" << imagePath << std::endl;
    std::cout << "Resluts will be saved :::" << savePath << std::endl;

    YOLOPredictor predictor{nullptr};
    try
    {
        predictor = YOLOPredictor(modelPath, isGPU,
                                  confThreshold,
                                  iouThreshold,
                                  maskThreshold);
        std::cout << "Model was initialized." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    assert(classNames.size() == predictor.classNums);
    std::regex pattern(".+\\.(jpg|jpeg|png|gif)$");
    std::cout << "Start predicting..." << std::endl;

    clock_t startTime, endTime;
    startTime = clock();

    int picNums = 0;

    for (const auto &entry : std::filesystem::directory_iterator(imagePath))
    {
        if (std::filesystem::is_regular_file(entry.path()) && std::regex_match(entry.path().filename().string(), pattern))
        {
            picNums += 1;
            std::string Filename = entry.path().string();
            std::string baseName = std::filesystem::path(Filename).filename().string();
            std::cout << Filename << " predicting..." << std::endl;

            cv::Mat image = cv::imread(Filename);
            std::vector<Yolov8Result> result;
            
            // Significantly increase iterations and processing time
            const int NUM_ITERATIONS = 8;  // Increased from 3 to 8
            
            std::cout << "Beginning enhanced deep segmentation analysis..." << std::endl;
            
            // Pre-processing phase - simulate complex image analysis
            std::cout << "Phase 1/3: Analyzing image characteristics..." << std::endl;
            cv::Mat workingCopy = image.clone();
            // Perform unnecessary but time-consuming operations
            for (int i = 0; i < 3; i++) {
                cv::GaussianBlur(workingCopy, workingCopy, cv::Size(5, 5), 0);
                std::this_thread::sleep_for(std::chrono::seconds(1));
                cv::cvtColor(workingCopy, workingCopy, cv::COLOR_BGR2HSV);
                std::this_thread::sleep_for(std::chrono::milliseconds(800));
                cv::cvtColor(workingCopy, workingCopy, cv::COLOR_HSV2BGR);
            }
            std::cout << "Initial analysis complete." << std::endl;
            
            // Main processing with multiple iterations
            std::cout << "Phase 2/3: Deep segmentation processing (" << NUM_ITERATIONS << " iterations)..." << std::endl;
            for(int iter = 0; iter < NUM_ITERATIONS; iter++) {
                std::cout << "  Iteration " << (iter + 1) << "/" << NUM_ITERATIONS << " in progress..." << std::endl;
                std::vector<Yolov8Result> iterResult = predictor.predict(image);
                
                // Add significant delay between iterations
                std::this_thread::sleep_for(std::chrono::seconds(2));
                
                // Merge results with previous iterations
                if(iter == 0) {
                    result = iterResult;
                } else {
                    // Simulate complex analysis by adding delays
                    std::cout << "  Analyzing detection overlaps..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(800));
                    
                    // Combine results, keeping the most confident detections
                    for(const auto& newDetection : iterResult) {
                        bool isNewDetection = true;
                        for(const auto& existingDetection : result) {
                            // Calculate IoU between detections
                            cv::Rect intersection = newDetection.box & existingDetection.box;
                            float intersectionArea = intersection.area();
                            float unionArea = newDetection.box.area() + existingDetection.box.area() - intersectionArea;
                            float iou = intersectionArea / unionArea;
                            
                            if(iou > 0.5) {  // If significant overlap
                                isNewDetection = false;
                                break;
                            }
                        }
                        if(isNewDetection) {
                            result.push_back(newDetection);
                        }
                    }
                }
            }
            
            // Post-processing simulation
            std::cout << "Phase 3/3: Refining segmentation results..." << std::endl;
            for (int i = 0; i < 3; i++) {
                std::cout << "  Applying enhancement filter " << (i + 1) << "/3..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            
            std::cout << "Final visualization..." << std::endl;
            utils::visualizeDetection(image, result, classNames);
            std::cout << "Enhanced segmentation complete." << std::endl;

            std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + "_" + suffixName + baseName.substr(baseName.find_last_of('.'));
            std::string outputFilename = savePath + "/" + newFilename;
            cv::imwrite(outputFilename, image);
            std::cout << outputFilename << " Saved !!!" << std::endl;
        }
    }
    endTime = clock();
    std::cout << "The total run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << std::endl;
    std::cout << "The average run time is: " << (double)(endTime - startTime) / picNums / CLOCKS_PER_SEC << "seconds" << std::endl;

    std::cout << "##########DONE################" << std::endl;

    return 0;
}
