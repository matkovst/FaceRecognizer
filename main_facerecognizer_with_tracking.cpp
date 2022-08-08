#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "src/face_detector.h"
#include "src/face_extractor.h"
#include "src/box_tracker.h"
#include "src/renderer.h"
#include "src/math.h"
#include "src/face.h"

constexpr float DetectionNoise { 0.1f };
constexpr std::int64_t DetectionFrequency { 160 }; // msec

const std::string ProgramName { "FaceRecognizer" };
const std::string CommandLineParams =

    /* Main parameters */
    "{ help h usage ?    |      | print this message }"
    "{ @input i          |   0  | input video or stream }"
    "{ @persons_file p   |      | path to file with person embeddings }"
    "{ @detector_path d  |   ../../data/yolov5s-face.onnx   | path to face detection model }"
    "{ @recognizer_path r|   ../../data/adaface_ir18_vgg2.torchscript   | path to face recognition model }"

    /* Auxilary parameters */
    "{ conf              |   0.25   | minimal detection confidence }"
    "{ sim_thr           |   0.25   | minimal similarity }"
    "{ gpu               |   0      | enable gpu }"
    "{ input_scale       |   1.0    | input resolution scale }"
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
    const auto personsFile = parser.get<std::string>("@persons_file");
    const auto detectorPath = parser.get<std::string>("@detector_path");
    const auto recognizerPath = parser.get<std::string>("@recognizer_path");
    const auto minConfidence = parser.get<float>("conf");
    const auto minSimilarity = parser.get<float>("sim_thr");
    const auto enableGpu = static_cast<bool>(parser.get<int>("gpu"));
    const auto inputScale = parser.get<float>("input_scale");
    
    /* Fetch existing embeddings from disk */
    std::vector<std::string> personNames;
    Matr personEmbeddings;
    if (!personsFile.empty())
    {
        cv::FileStorage personsFileStorage;
        try
        {
            personsFileStorage.open(personsFile, cv::FileStorage::READ);
            if (!personsFileStorage.isOpened())
            {
                std::cerr << "Failed to open -persons_file" << std::endl;
                return EXIT_FAILURE;
            }
        }
        catch(const cv::Exception& e)
        {
            std::cerr << "Failed to open -persons_file:\n" << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        const auto personNamesNode = personsFileStorage["Names"];
        if (cv::FileNode::SEQ != personNamesNode.type())
        {
            std::cerr << "Failed to read names from -persons_file. Data invalid." << std::endl;
            return EXIT_FAILURE;
        }

        for (auto it = personNamesNode.begin(); it != personNamesNode.end(); ++it)
            personNames.emplace_back(static_cast<std::string>(*it));
        personEmbeddings.reserve(personNames.size());
        for (const auto& name : personNames)
        {
            cv::Mat embeddingMat;
            personsFileStorage[name] >> embeddingMat;
            std::vector<float> embedding(embeddingMat.begin<float>(), embeddingMat.end<float>());
            personEmbeddings.emplace_back(std::move(embedding));
        }
        std::cout << "Loaded " << personEmbeddings.size() << " persons from disk" << std::endl;
    }
    
    /* Initialize general stuff */
    FaceDetector faceDetector(detectorPath, enableGpu);
    FaceExtractor faceExtractor(recognizerPath, enableGpu);
    BoxTracker boxTracker(DetectionNoise);
    PeriodicTrigger trigger(DetectionFrequency);

    /* Capture input */
    cv::VideoCapture capture;
    if ("0" == input)
        capture.open(0);
    else
        capture.open(input);
    if (!capture.isOpened())
    {
        std::cerr << "Could not open video" << std::endl;
        return EXIT_FAILURE;
    }
    const double fps = std::clamp(capture.get(cv::CAP_PROP_FPS), 1.0, 30.0);

    /* Start main loop */
    std::int64_t frameNum = 1;
    for (;; ++frameNum)
    {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
            break;

        if (1.0 != inputScale)
            cv::resize(frame, frame, cv::Size(), inputScale, inputScale);

        const std::int64_t timestamp = (frameNum / fps) * 1000;

        /* NN magic */

        // 1. Detect face with given frequency
        FaceDetector::DetectionResult faceDetectionResult;
        const bool rocknroll = trigger.rocknroll(timestamp);
        if (rocknroll)
        {
            const auto faceDetectionResults = faceDetector.detect(frame, minConfidence);
            if (faceDetectionResults.size() > 0)
                faceDetectionResult = faceDetectionResults[0];
        }

        // 2. Keep tracking the face
        cv::Rect faceTracklet;
        if (!boxTracker.initialized())
        {
            boxTracker.init(faceDetectionResult.boundingBox);
            faceTracklet = faceDetectionResult.boundingBox;
        }
        else
        {
            faceTracklet = boxTracker.update(faceDetectionResult.boundingBox);
        }

        std::vector<Face> faces;
        faces.emplace_back(
            faceTracklet,
            faceDetectionResult.landmarks,
            faceDetectionResult.confidence,
            -1,
            "unknown",
            frame(faceTracklet).clone(),
            -1.0f
            );
        auto& face = faces.at(0);
        
        // 2. Extract face embedding and identify it
        
        // 2.1. Extract & idenfity (try #1)
        auto faceEmbedding = faceExtractor.extract(face.crop);
        auto [bestId, bestSim] = searchMostSimilarEmbedding(personEmbeddings, faceEmbedding);

        // // 2.3. Extract & idenfity (try #2 on aligned face) if the first try failed
        // if (minSimilarity > bestSim)
        // {
        //     const cv::Mat alignedFaceCrop = alignFace2(
        //         frame, 
        //         face.boundingBox, 
        //         face.landmarks, 
        //         face.boundingBox.size(), 
        //         FaceExtractor::ReferencePoints2);

        //     faceEmbedding = faceExtractor.extract(alignedFaceCrop);
        //      std::tie(bestId, bestSim) = searchMostSimilarEmbedding(personEmbeddings, faceEmbedding);
        // }

        if (bestSim >= minSimilarity)
        {
            face.nameId = bestId;
            face.name = personNames[bestId];
            face.similarity = bestSim;
        }

        /* Render results */
        renderFaces(frame, faces);
        const auto color = (rocknroll) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        cv::rectangle(frame, faceTracklet, color, 2);
        cv::imshow(ProgramName, frame);

        const auto key = static_cast<char>(cv::waitKey(15));
        if (27 == key || 'q' == key)
            break;
    }

    capture.release();
    cv::destroyAllWindows();

    std::cout << "Program successfully finished" << std::endl;
    return EXIT_SUCCESS;
}