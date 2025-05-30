#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include "cmdline.h"
#include "utils.h"
#include "yolov8Predictor.h"
#include "nsgs/NsgsPredictor.h"
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
    cmd.add("nsgs", '\0', "Use NSGS approach (Neuro-Scheduling for Graph Segmentation).");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    bool useNSGS = cmd.exist("nsgs");
    
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
    
    // Create the appropriate predictor based on user selection
    if (useNSGS) {
        std::cout << "Using NSGS approach for processing" << std::endl;
        
        // Initialize NSGS predictor
        NsgsPredictor* predictor = nullptr;
        try
        {
            predictor = new NsgsPredictor(modelPath, isGPU,
                                         confThreshold,
                                         iouThreshold,
                                         maskThreshold);
            std::cout << "NSGS Model was initialized." << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
            return -1;
        }
        
        assert(classNames.size() == predictor->classNums);
        std::regex pattern(".+\\.(jpg|jpeg|png|gif)$");
        std::cout << "Start predicting with NSGS..." << std::endl;

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
                std::cout << Filename << " predicting with NSGS..." << std::endl;

                cv::Mat image = cv::imread(Filename);
                if (image.empty()) {
                    std::cerr << "Error: Could not read image " << Filename << std::endl;
                    continue;
                }
                
                std::vector<Yolov8Result> result;
                
                try {
                    // Use the detect method directly to get better timeout handling
                    result = predictor->detect(image, confThreshold, iouThreshold, maskThreshold);
                    
                    // Always process whatever results we have, even if partial
                    if (!result.empty()) {
                        std::cout << "NSGS: Got " << result.size() << " detection results" << std::endl;
                        
                        // Visualize the detection on the image
                        utils::visualizeDetection(image, result, classNames);
                        
                        // Save the image with detections
                        std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + "_" + suffixName + baseName.substr(baseName.find_last_of('.'));
                        std::string outputFilename = savePath + "/" + newFilename;
                        bool saved = cv::imwrite(outputFilename, image);
                        
                        if (saved) {
                            std::cout << outputFilename << " Saved !!!" << std::endl;
                        } else {
                            std::cerr << "Error saving to " << outputFilename << std::endl;
                        }
                    } else {
                        std::cout << "NSGS: No detection results for " << Filename << std::endl;
                    }
                } catch (const std::exception &e) {
                    std::cerr << "Error processing " << Filename << ": " << e.what() << std::endl;
                }
            }
        }
        endTime = clock();
        std::cout << "The total run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << std::endl;
        std::cout << "The average run time is: " << (double)(endTime - startTime) / picNums / CLOCKS_PER_SEC << "seconds" << std::endl;
        delete predictor;
    }
    else {
        // Use the original YOLOPredictor
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
                std::vector<Yolov8Result> result = predictor.predict(image);
                
                utils::visualizeDetection(image, result, classNames);

                std::string newFilename = baseName.substr(0, baseName.find_last_of('.')) + "_" + suffixName + baseName.substr(baseName.find_last_of('.'));
                std::string outputFilename = savePath + "/" + newFilename;
                cv::imwrite(outputFilename, image);
                std::cout << outputFilename << " Saved !!!" << std::endl;
            }
        }
        endTime = clock();
        std::cout << "The total run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "seconds" << std::endl;
        std::cout << "The average run time is: " << (double)(endTime - startTime) / picNums / CLOCKS_PER_SEC << "seconds" << std::endl;
    }

    std::cout << "##########DONE################" << std::endl;

    return 0;
}
