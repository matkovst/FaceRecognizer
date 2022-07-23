#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "src/face_detector.h"
#include "src/face_extractor.h"
#include "src/math.h"

const std::string ProgramName { "FaceCollector" };
const std::string CommandLineParams =
    "{ help h usage ?    |      | print this message }"
    "{ @input i          |      | path to input photos }"
    "{ @output o         |      | path to output file with embeddings }"
    "{ @detector_path d  |   ../../data/yolov5s-face.onnx   | path to face detection model }"
    "{ @recognizer_path r|   ../../data/adaface_ir18_vgg2.torchscript   | path to face recognition model }"
    ;

int main(int argc, char *argv[])
{
    std::cout << "Program started" << std::endl;

    /* Check and parse cmd args */
    cv::CommandLineParser parser(argc, argv, CommandLineParams);
    parser.about(ProgramName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }
    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }
    const auto input = parser.get<std::string>("@input");
    const auto output = parser.get<std::string>("@output");
    const auto detectorPath = parser.get<std::string>("@detector_path");
    const auto recognizerPath = parser.get<std::string>("@recognizer_path");

    if (input.empty())
    {
        std::cerr << "You must specify -input" << std::endl;
        return EXIT_FAILURE;
    }
    if (output.empty())
    {
        std::cerr << "You must specify -output" << std::endl;
        return EXIT_FAILURE;
    }
    
    /* Initialize general stuff */
    FaceDetector faceDetector(detectorPath);
    FaceExtractor faceExtractor(recognizerPath);

    /* Iterate over input directory and extracting embeddings from internal photos */
    cv::FileStorage fileStorage(output, cv::FileStorage::WRITE);
    std::vector<std::string> personNames;
    for (const auto& personDirEntry : fs::directory_iterator(input))
    {
        if (!fs::is_directory(personDirEntry))
            continue;

        const auto personName = personDirEntry.path().filename().string();
        std::vector<std::vector<float>> personEmbeddings;
        for (const auto & personPhotosEntry : fs::directory_iterator(personDirEntry))
        {
            if (!fs::is_regular_file(personPhotosEntry))
                continue;

            const auto ext = personPhotosEntry.path().extension();
            if (!(ext == ".png" || ext == ".jpg" || ext == ".jpeg"))
                continue;

            const auto photoAbsPath = fs::absolute(personPhotosEntry.path()).string();
            const cv::Mat photo = cv::imread(photoAbsPath);
            if (photo.empty())
                continue;

            const auto faceDetectionResults = faceDetector.detect(photo);
            if (faceDetectionResults.empty())
                continue;

            const auto& faceBoundingBox = faceDetectionResults[0].boundingBox;
            const auto faceCrop = photo(faceBoundingBox);
            auto faceEmbedding = faceExtractor.extract(faceCrop);
            if (faceEmbedding.empty())
                continue;

            personEmbeddings.emplace_back(std::move(faceEmbedding));
        }

        /* Write embedding to disk */
        if (personEmbeddings.size() > 0)
        {
            const auto personAvgEmbedding = avgEmbedding(personEmbeddings);
            fileStorage << personName 
                << cv::Mat(1, personAvgEmbedding.size(), CV_32F, (void*)personAvgEmbedding.data());
            personNames.emplace_back(personName);

            std::cout << "Embeddings extracted for " << personName << std::endl;
        }
    }
    if (personNames.size() > 0)
    {
        fileStorage << "Names" << "[";
        for (const auto& name : personNames)
            fileStorage << name;
        fileStorage << "]";
    }

    std::cout << "Program successfully finished" << std::endl;
    return EXIT_SUCCESS;
}