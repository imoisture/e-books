FAQ
==========================

编译安装
----------------------------

Q：安装依赖或源码编译过程中出现 ``unrecognized command line option xxx`` 类似报错，例如：

   ::
   
     "unrecognized command line option ‘-fno-plt’"
     "unrecognized command line option ‘c++11’"
     "unrecognized command line option ‘c++14’"

A：由于环境中g++版本较低导致无法识别指定的编译参数，建议将g++版本升级为大于等于5.4.0。

Q：执行 ``python setup.py install`` 安装Torchvision出现编译报错：

   ::
   
      error: cannot convert 'std::nullptr_t' to 'Py_ssize_t {aka long int}' in initialization.

A：安装Torchvision依赖FFmpeg，需要使用源码安装FFmpeg。或者使用pip安装 ``pip install torchvision==0.7.0`` 。

Q: 为什么无法运行Torchvision的自定义算子？

A: 如果希望使用Torchvision中的第三方算子，例如需要在MLU设备上运行NMS算子，此时需要安装与Cambricon PyTorch版本对应的Cambricon Torchvision。
若仅需要使用Torchvision的网络模型，可以使用 ``pip install torchvision==0.7.0`` 安装原生框架的Torchvision。

Q: Ubuntu16.04系统在AArch64平台交叉编译过程中出现报错：``fatal error: 'sys/cdefs.h' file not found``。

A: 执行命令 ``sudo apt-get install libc6-dev-i386`` 安装依赖项。


Cambricon PyTorch融合推理
----------------------------

Q：原位 ``net = torch::jit::trace(net, example_forward_input)`` 后的net执行 ``net.to('mlu')`` 时，报如下错误：

   ::
   
     Traceback (most recent call last):
     File "test.py", line 20, in <module>
        net.to("mlu")
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 608, in to
        return self._apply(convert)
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 354, in _apply
        module._apply(fn)
     File "venv/pytorch/lib/python3.6/site-packages/torch/nn/modules/module.py", line 382, in _apply
        assert isinstance(param, Parameter)
     AssertionError

A：由于trace后将net原位赋值，导致 ``net.parameters()`` 中param的类型由 ``torch.nn.parameter.Parameter`` 变为 ``torch.Tensor``，
进而导致 ``assert isinstance(param, Parameter)`` 失败。要解决该问题，使用非原位的方式执行trace，
即：``traced_net = torch::jit::trace(net, example_forward_input)``。

Q：为什么上述问题中使用原位方式trace时，执行 ``net.to('cpu')`` 未报错？

A：原生 ``to`` 使用以下接口控制是否覆盖Tensor：

   ::

     torch.__future__.set_overwrite_module_params_on_conversion(bool flag) 

由于原生CPU在执行 ``to`` 操作时，默认行为是直接替换Tensor中的TensorImpl结构，即不覆盖Tensor，因此未报错。
而MLU的MLUTensorImpl为TensorImpl子类，不能原位直接替换，而是使用覆盖Tensor的方式，因此会报错。

.. attention::

   | 目前Cambricon PyTorch仅支持 ``torch.__future__.set_overwrite_module_params_on_conversion(True)`` 条件下，CPU执行不报错的情况。


C++前端使用
----------------------------
Q: 程序运行过程中出现运行中断，报错信息为：

   ::
   
     "PyTorch is not linked with support for mlu devices"
   
     "The running of the task requires MLU device initialization."

A：程序编写没有调用 ``torch_mlu::torch_mlu_init()`` 进行设备初始化。在使用所有C++前端接口前，必须调用该函数初始化MLU设备。

Q：使用Torchvision源码安装C++前端过程中安装出现报错：

   ::
   
     By not providing "FindPython3.cmake" in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by "Python3", but CMake did not find one.

A：Torchvision源码安装C++前端过程中安装依赖cmake版本，推荐cmake版本为 >= 3.12.0。

Q：使用Torchvision源码安装C++前端过程中安装出现报错：

   ::
   
      By not providing "FindTorchVision.cmake" in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by "TorchVision", but CMake did not find one.

A：Torchvision源码编译安装依赖Torch依赖，需要指定 ``CMAKE_PREFIX_PATH`` 添加Torch依赖。

Q：为什么模型训练迭代过程中数据存在波动，而且程序中包含种子设置torch::manual_seed()？

A：MLU支持算子硬件加速，对个别算子输出会不唯一。常见算子：torch::conv反向输出梯度，dropout训练模型随机不受seed控制。

Q: 使用 ``catch/script/release/independent_build.sh`` 脚本打patch出现告警：
   
   ::

      Warning: You have applied patches to Pytorch.

A: 该告警说明用户已经打过patch到PyTorch，无需重复操作。

Q: 使用MagicMind wheel包作为依赖进行编译和运行，如何解决兼容性问题？

A: 目前已知的可能出现的兼容性问题有2个：

   1.pandas安装问题：catch requirements.txt中指定pandas==0.22.0,该版本在python3.7环境下可能会无法兼容，可通过安装高版本pandas（比如1.1.5）来解决。

   2.需要额外把llvm的依赖包路径加入到LD_LIBRARY_PATH中去。 假设你的MagicMind wheel是old-abi制作出来的，则按照如下方法设置：

     :: shell

         export LD_LIBRARY_PATH=YOUR_LOCAL_PATH/neuware_home/lib64:YOUR_LOCAL_PATH/neuware_home/lib/llvm-mm-cxx11-old-abi/lib/:/usr/lib/python3.7/site-packages/magicmind
