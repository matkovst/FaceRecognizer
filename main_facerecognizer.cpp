#include <cstdlib>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "src/face_detector.h"
#include "src/face_extractor.h"
#include "src/renderer.h"
#include "src/math.h"
#include "src/face.h"
#include "data/test_embeddings.h"

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

        /* NN magic */

        // 1. Detect faces
        const auto faceDetectionResults = faceDetector.detect(frame, minConfidence);

        std::vector<Face> faces;
        faces.reserve(faceDetectionResults.size());
        for (int i = 0; i < faceDetectionResults.size(); ++i)
            faces.emplace_back(
                faceDetectionResults[i].boundingBox,
                faceDetectionResults[i].landmarks,
                faceDetectionResults[i].confidence,
                -1,
                "unknown",
                frame(faceDetectionResults[i].boundingBox).clone(),
                -1.0f
                );
        
        // 2. Extract face embeddings and identify them
        for (int i = 0; i < faces.size(); ++i)
        {
            // 2.1. Extract & idenfity (try #1)
            auto faceEmbedding = faceExtractor.extract(faces[i].crop);
            auto [bestId, bestSim] = searchMostSimilarEmbedding(personEmbeddings, faceEmbedding);

            // 2.3. Extract & idenfity (try #2 on aligned face) if the first try failed
            if (minSimilarity > bestSim)
            {
                const cv::Mat alignedFaceCrop = alignFace2(
                    frame, 
                    faces.at(i).boundingBox, 
                    faces.at(i).landmarks, 
                    faces.at(i).boundingBox.size(), 
                    FaceExtractor::ReferencePoints2);

                faceEmbedding = faceExtractor.extract(alignedFaceCrop);
                std::tie(bestId, bestSim) = searchMostSimilarEmbedding(personEmbeddings, faceEmbedding);
            }

            if (bestSim >= minSimilarity)
            {
                faces[i].nameId = bestId;
                faces[i].name = personNames[bestId];
                faces[i].similarity = bestSim;
            }
        }

        /* Render results */
        renderFaces(frame, faces);
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