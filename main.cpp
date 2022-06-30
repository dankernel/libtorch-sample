/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  libtorch sample code
 *
 *        Version:  1.0
 *        Created:  2022년 06월 29일 23시 44분 16초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Junhyung Park (dankernel), 
 *   Organization:  dankernel.sciomagelab.com
 *
 * =====================================================================================
 */
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[]) {

    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module> <iamge-file>\n";
        return -1;
    }

    cv::Mat img_bgr_u8 = cv::imread(argv[2], cv::IMREAD_COLOR);

    // Resize
    cv::resize(img_bgr_u8, img_bgr_u8, cv::Size(224, 224));

    // BRG2RGB
    cv::Mat img_rgb_u8;
    cv::cvtColor(img_bgr_u8, img_rgb_u8, cv::COLOR_BGR2RGB);

    // To tensor
    torch::Tensor input_tensor = torch::from_blob(img_rgb_u8.data, {224, 224, 3}, torch::kByte);
    input_tensor = input_tensor.permute({2, 0, 1});
    input_tensor = input_tensor.toType(torch::kFloat);
    input_tensor = input_tensor.div(255);
    input_tensor = input_tensor.unsqueeze(0);

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Inference
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // Print result
    std::cout << torch::argmax(torch::softmax(output, 1)) << '\n';
    std::cout << torch::argmax(output) << '\n';

    std::cout << "ok\n";

    return 0;
}
