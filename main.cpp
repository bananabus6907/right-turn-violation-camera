#include <opencv2/opencv.hpp>
#include <torch/script.h> // Libtorch
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <deque>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

// 1. YOLO4Model Class: YOLOv4 모델 로드 및 추론 (OpenCV 기반)
class YOLO4Model {
public:
    cv::dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
    std::vector<std::string> classLabels;

    YOLO4Model(const std::string& modelPath, const std::string& configPath, const std::vector<std::string>& labels,
               float confThreshold, float nmsThreshold)
        : classLabels(labels), confidenceThreshold(confThreshold), nmsThreshold(nmsThreshold) {
        if (!std::filesystem::exists(modelPath) || (!configPath.empty() && !std::filesystem::exists(configPath))) {
            throw std::runtime_error("Error: Model or config file does not exist at specified path.");
        }

        net = configPath.empty() ? cv::dnn::readNet(modelPath) : cv::dnn::readNet(modelPath, configPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }

    std::vector<cv::Rect> detectObjects(const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences) {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(640, 640), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        std::vector<cv::Rect> boxes;
        for (const auto& output : outputs) {
            const float* data = reinterpret_cast<const float*>(output.data);
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                float confidence = data[4];
                if (confidence >= confidenceThreshold) {
                    cv::Mat scores = output.row(i).colRange(5, output.cols);
                    cv::Point classIdPoint;
                    double maxClassScore;
                    cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

                    if (maxClassScore > confidenceThreshold) {
                        int centerX = static_cast<int>(data[0] * frame.cols);
                        int centerY = static_cast<int>(data[1] * frame.rows);
                        int width = static_cast<int>(data[2] * frame.cols);
                        int height = static_cast<int>(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        boxes.push_back(cv::Rect(left, top, width, height));
                        confidences.push_back(static_cast<float>(maxClassScore));
                        classIds.push_back(classIdPoint.x);
                    }
                }
            }
        }
        return boxes;
    }
};

// 2. YOLO5Model Class: YOLOv5 모델 로드 및 추론 (PyTorch 기반)
class YOLO5Model {
public:
    torch::jit::script::Module module; // TorchScript 모델
    std::vector<std::string> classLabels;
    float confidenceThreshold;
    float nmsThreshold;
    torch::Device device;

    // 생성자 정의
    YOLO5Model(const std::string& modelPath, const std::vector<std::string>& labels,
               float confThreshold, float nmsThreshold)
        : classLabels(labels), confidenceThreshold(confThreshold), nmsThreshold(nmsThreshold),
          device(torch::kCPU) { // CPU로 고정

        // 모델 파일 확인
        if (!std::filesystem::exists(modelPath)) {
            throw std::runtime_error("Error: Model file does not exist at specified path.");
        }

        std::cout << "Using device: CPU" << std::endl;

        // 모델 로드 및 CPU 이동
        module = torch::jit::load(modelPath);
        module.to(device);
        module.eval();
    }

    std::vector<cv::Rect> detectObjects(const cv::Mat& frame, std::vector<int>& classIds, std::vector<float>& confidences) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // Tensor 생성: CPU 전용
        auto inputTensor = torch::from_blob(resized.data, {1, resized.rows, resized.cols, 3}, torch::kByte)
                               .to(torch::kFloat)
                               .permute({0, 3, 1, 2}) // [H, W, C] -> [N, C, H, W]
                               .div(255.0)
                               .to(device);

        // 모델 추론
        auto outputs = module.forward({inputTensor}).toTensor().squeeze(0).to(device);

        std::vector<cv::Rect> boxes;
        for (int i = 0; i < outputs.size(0); ++i) {
            auto row = outputs[i];
            float confidence = row[4].item<float>();

            if (confidence >= confidenceThreshold) {
                auto scores = row.slice(5, row.size(0));
                auto maxResult = scores.max(0);

                float classConfidence = std::get<0>(maxResult).item<float>();
                int classId = std::get<1>(maxResult).item<int>();

                if (classConfidence >= confidenceThreshold) {
                    int centerX = static_cast<int>(row[0].item<float>() * frame.cols);
                    int centerY = static_cast<int>(row[1].item<float>() * frame.rows);
                    int width = static_cast<int>(row[2].item<float>() * frame.cols);
                    int height = static_cast<int>(row[3].item<float>() * frame.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(classConfidence);
                    classIds.push_back(classId);
                }
            }
        }
        return boxes;
    }
};



// 3. ViolationDetector Class: 교차로 위반 감지
class ViolationDetector {
public:
    bool detectViolation(const std::vector<cv::Rect>& pedestrianCrossings, const std::vector<cv::Rect>& vehicles,
                         const std::string& signalState) {
        for (const auto& crossing : pedestrianCrossings) {
            for (const auto& vehicle : vehicles) {
                if (signalState == "초록불" && vehicle.x < crossing.x + crossing.width) {
                    return true; // 교차로 위반 감지
                }
            }
        }
        return false; // 위반 없음
    }
};

// 4. FrameProcessor Class: 프레임 처리 및 시각화
class FrameProcessor {
public:
    void drawBoxes(const cv::Mat& inputFrame, const std::vector<cv::Rect>& boxes,
               const std::vector<int>& classIds, const std::vector<float>& confidences,
               const std::vector<std::string>& classLabels, const std::string& text = "",
               const cv::Scalar& color = cv::Scalar(0, 255, 0)) {
        cv::Mat frame = inputFrame.clone(); // 복사본 생성
        for (size_t i = 0; i < boxes.size(); ++i) {
            cv::rectangle(frame, boxes[i], color, 2);
            std::string label = classLabels[classIds[i]] + ": " + std::to_string(confidences[i]);
            cv::putText(frame, label, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        // 필요시 후처리 추가
    }

};

// 5. TrafficViolationSystem Class: 배치 처리 추가 및 교차로 위반 감지
class TrafficViolationSystem {
private:
    YOLO4Model yolo4Model;
    YOLO5Model yolo5Model;
    FrameProcessor frameProcessor;
    ViolationDetector violationDetector;

    static const int BATCH_SIZE = 16; // 배치 크기
    std::deque<cv::Mat> frameBuffer; // 배치 버퍼

public:
    TrafficViolationSystem(const YOLO4Model& yolo4, const YOLO5Model& yolo5)
        : yolo4Model(yolo4), yolo5Model(yolo5) {}

    void processVideo(const std::string& videoPath, const std::string& outputPath) {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video file." << std::endl;
            return;
        }

        cv::VideoWriter out(outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            cap.get(cv::CAP_PROP_FPS),
                            cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        cv::Mat frame;
        while (cap.read(frame)) {
            frameBuffer.push_back(frame.clone()); // 프레임 버퍼에 추가

            if (frameBuffer.size() == BATCH_SIZE) {
                processBatch(out); // 배치 처리
                frameBuffer.clear(); // 배치 초기화
            }

            if (cv::waitKey(1) == 'q') break;
        }

        // 남은 프레임 처리
        if (!frameBuffer.empty()) {
            processBatch(out);
        }

        cap.release();
        out.release();
        cv::destroyAllWindows();
    }

private:
    void processBatch(cv::VideoWriter& out) {
        for (const auto& frame : frameBuffer) {
            std::vector<int> classIds4, classIds5;
            std::vector<float> confidences4, confidences5;

            // YOLO4와 YOLO5 추론
            auto boxes4 = yolo4Model.detectObjects(frame, classIds4, confidences4);
            auto boxes5 = yolo5Model.detectObjects(frame, classIds5, confidences5);

            // 바운딩 박스 스케일 변환 및 후처리
            scaleBoundingBoxes(boxes4, frame.size());
            scaleBoundingBoxes(boxes5, frame.size());

            // 신호등 상태와 교차로 위반 검출
            std::vector<cv::Rect> pedestrianCrossings;
            std::string signalState = "빨간불";

            for (size_t i = 0; i < boxes5.size(); ++i) {
                if (classIds5[i] == 0) { // 횡단보도
                    pedestrianCrossings.push_back(boxes5[i]);
                } else if (classIds5[i] == 1) { // 빨간불
                    signalState = "빨간불";
                } else if (classIds5[i] == 2) { // 초록불
                    signalState = "초록불";
                }
            }

            bool violation = violationDetector.detectViolation(pedestrianCrossings, boxes4, signalState);

            // 시각화
            frameProcessor.drawBoxes(frame, boxes4, classIds4, confidences4, yolo4Model.classLabels);
            frameProcessor.drawBoxes(frame, boxes5, classIds5, confidences5, yolo5Model.classLabels,
                                      violation ? "[!위반]" : "", violation ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0));

            // 결과 저장
            out.write(frame);
            cv::imshow("Traffic Monitoring", frame);
        }
    }

    void scaleBoundingBoxes(std::vector<cv::Rect>& boxes, const cv::Size& frameSize) {
        for (auto& box : boxes) {
            box.x = static_cast<int>(box.x * frameSize.width / 640.0);
            box.y = static_cast<int>(box.y * frameSize.height / 640.0);
            box.width = static_cast<int>(box.width * frameSize.width / 640.0);
            box.height = static_cast<int>(box.height * frameSize.height / 640.0);
        }
    }
};

// Main Function
int main() {
    std::vector<std::string> labels = {"차량", "번호판", "횡단보도", "빨간불", "초록불"};
    std::vector<std::string> yolov4Labels = {"차량", "번호판"};
    std::vector<std::string> yolov5Labels = {"횡단보도", "빨간불", "초록불"};

   YOLO4Model yolo4(
    "/home/airobot/opensource/yolov4/yolov4-ANPR.weights",
    "/home/airobot/opensource/yolov4/yolov4-ANPR.cfg",
    yolov4Labels, // YOLOv4에만 해당 라벨 전달
    0.5, 0.3
);

YOLO5Model yolo5(
    "/home/airobot/opensource/yolov5/best.torchscript",
    yolov5Labels, // YOLOv5에만 해당 라벨 전달
    0.5, 0.45
);





    TrafficViolationSystem system(yolo4, yolo5);
    system.processVideo("/home/airobot/opensource/data/test.mp4", "/home/airobot/opensource/output/output.mp4");

    return 0;
} 