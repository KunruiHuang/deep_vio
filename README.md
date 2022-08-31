# pytorch模型c++推理
+ test_data:测试数据
+ checkpoints: 模型
    + c++:保存经过torch::jit::trace转换过的可供c++API调用的pytorch模型
    + python:训练后直接保存下来的参数字典
+ src:测试源码

### 依赖:
```bash
opencv:3.4.3
libtorch:1.8.2 + cuda10.2
```
使用方法：


编译
```bash
mkdir build
cd build
cmake ..
make
```
运行

```bash
./TorchDemo ../checkpoints/c++/mnist_cnn_cc1.pt ../test_data/data.csv
