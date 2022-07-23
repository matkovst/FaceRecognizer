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
    "{ help h usage ?    |      | print this message }"
    "{ @input i          |   0   | input video or stream }"
    "{ @persons_file p   |       | path to file with person embeddings }"
    "{ @detector_path d  |   ../../data/yolov5s-face.onnx   | path to face detection model }"
    "{ @recognizer_path r|   ../../data/adaface_ir18_vgg2.torchscript   | path to face recognition model }"
    "{ conf              |   0.25   | minimal detection confidence }"
    "{ sim_thr           |   0.25   | minimal similarity }"
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
    FaceDetector faceDetector(detectorPath);
    FaceExtractor faceExtractor(recognizerPath);

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

        /* NN magic */

        // 1. Detect faces
        const auto faceDetectionResults = faceDetector.detect(frame, minConfidence);
        
        // 2. Extract face embeddings
        std::vector<cv::Mat> faceImages;
        faceImages.reserve(faceDetectionResults.size());
        for (const auto& faceDetectionResult : faceDetectionResults)
            faceImages.emplace_back(frame(faceDetectionResult.boundingBox).clone());
        const auto faceEmbeddings = faceExtractor.extract(faceImages);

        // 3. Identify faces
        std::vector<Face> faces;
        faces.reserve(faceEmbeddings.size());
        for (int i = 0; i < faceEmbeddings.size(); ++i)
        {
            int nameId = -1;
            float sim = -1.0f;
            std::string name = "unknown";
            if (personEmbeddings.size() > 0)
            {
                const auto [bestId, bestSim] = mostSimilar(personEmbeddings, faceEmbeddings[i]);
                if (bestSim >= minSimilarity)
                {
                    nameId = bestId;
                    sim = bestSim;
                    name = personNames[bestId];
                }
            }

            faces.emplace_back(
                faceDetectionResults[i].boundingBox,
                faceDetectionResults[i].landmarks,
                faceDetectionResults[i].confidence,
                nameId,
                std::move(name),
                sim
                );
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