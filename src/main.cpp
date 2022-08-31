#include <opencv2/imgproc/types_c.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/* main */
int main(int argc, const char *argv[]) {
  //接受一个运行参数
  // 1. 模型路径
  // 2. 储存imu数据的csv文件
  if (argc < 3) {
    std::cerr << "usage: CppProject <path-to-exported-script-module> "
              << "<path-to-imu_data>\n";
    return -1;
  }
  torch::jit::script::Module module = torch::jit::load(argv[1]);
  std::cout << "load model ok\n";
  FILE *fp;
  fp = fopen(argv[2], "r");
  // std::vector<float> imu_data_raw;
  // std::vector<float> imu_data_once;
  int cnt = 0;
  float imu_data[6000] = {0.0};
  int idx = 0;
  while (!feof(fp)) {
    float data[6] = {0.0};
    float time;
    fscanf(fp, "%f,%f,%f,%f,%f,%f,%f", &time, &data[0], &data[1], &data[2],
           &data[3], &data[4], &data[5]);
    for (int i = 0; i < 5; i++) imu_data[idx++] = data[i];
    cnt++;
    if (cnt >= 1000) break;
  }
  std::cout << "cnt:" << cnt << std::endl;
  auto imu_tensor = torch::from_blob(imu_data, {1, 10, 6, 100}, torch::kFloat);
  imu_tensor.print();
  std::cout << "imu_tensor.dtype()" << imu_tensor.dtype() << std::endl;
  std::vector<torch::jit::IValue> inputs;
  torch::DeviceType device_type = at::kCUDA;
  inputs.emplace_back(imu_tensor.to(device_type));
  auto outputs = module.forward(inputs).toTuple();
  std::cout << outputs->elements().size() << std::endl;
  //一个是速度，一个是协方差
  torch::Tensor out1 = outputs->elements()[0].toTensor().to(at::kCPU);
  torch::Tensor out2 = outputs->elements()[1].toTensor().to(at::kCPU);
  std::vector<float> v(out1.data_ptr<float>(),
                       out1.data_ptr<float>() + out1.numel());
  std::vector<float> covariance(out2.data_ptr<float>(),
                                out2.data_ptr<float>() + out2.numel());
  std::cout << v.size() << std::endl;
  std::cout << covariance.size() << std::endl;
  return 0;
}