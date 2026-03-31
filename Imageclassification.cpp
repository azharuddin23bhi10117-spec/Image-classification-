#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to load class labels
vector<string> loadLabels(const string& path) {
    vector<string> labels;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

int main() {
    // Load model
    string modelPath = "resnet50.onnx";
    Net net = readNetFromONNX(modelPath);

    // Load labels
    vector<string> labels = loadLabels("labels.txt");

    // Load image
    Mat image = imread("test.jpg");
    if (image.empty()) {
        cout << "Could not load image!" << endl;
        return -1;
    }

    // Preprocess image
    Mat blob;
    blobFromImage(image, blob, 1.0/255, Size(224, 224), Scalar(0,0,0), true, false);

    // Set input to the network
    net.setInput(blob);

    // Forward pass
    Mat output = net.forward();

    // Find class with highest score
    Point classIdPoint;
    double maxVal;
    minMaxLoc(output.reshape(1,1), 0, &maxVal, 0, &classIdPoint);

    int classId = classIdPoint.x;

    // Print result
    cout << "Predicted class: " << labels[classId] << endl;
    cout << "Confidence: " << maxVal << endl;

    return 0;
}
