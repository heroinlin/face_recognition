import torch
import torch.functional as F
from backbones import init_backbone
import sys
import os
from collections import OrderedDict
from utils.combine_conv_bn import fuse_module
"""
pytorch 模型转换脚本, 提供以下几种转化方式
1. pth  转 onnx  (0.4.0以下)
2. pth  转 onnx  (0.4.0以上)
3. pth  转 jit.pth  (1.0.0以上)
4. 模型转换时是否融合conv和bn(此处主要看保存模型本身是否融合)
5. 转换是否使用fp16

具体参数设置如下
    transform_model.py  <src_model_path> <convert_model_path> <convert_type>[1-3] {<combine_conv_bn>[0,1] <use_fp16>[0, 1]}
    transform_model.py  <原始模型路径> <转换后模型路径> <转换方式>[1-3] {<融合conv和bn>[0,1] <使用fp16>[0, 1]}
注：1.{} 内为可选参数, 默认值为0, 不开启
    2. 参数的最小个数为3个, 只能按顺序添加, 即第4个参数必须是<融合conv和bn>, 第5个为<使用fp16>
"""


def pytorch_version_to_0_3_1():
    """
    pytorch0.3.1导入0.4.1以上版本模型时加入以下代码块,可对比查看_utils.py文件修正相似错误,
   错误类型为(AttributeError: Can't get attribute '_rebuild_tensor_v2' on
   <module 'torch._utils' from '<pytorch0.3.1>\lib\site-packages\torch\_utils.py'>)
    Returns
    -------
    """
    #  使用以下函数代替torch._utils中的函数(0.3.1中可能不存在或者接口不同导致的报错)
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    try:
        torch._utils._rebuild_parameter
    except AttributeError:
        def _rebuild_parameter(data, requires_grad, backward_hooks):
            param = torch.nn.Parameter(data, requires_grad)
            param._backward_hooks = backward_hooks
            return param
        torch._utils._rebuild_parameter = _rebuild_parameter


def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = True
    elif isinstance(module, torch.nn.Upsample):
        module.align_corners = False
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn(module1)


def recursion_change_bn1(module):
    print(type(module))
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = True
    elif isinstance(module, torch.nn.Upsample):
        del module.align_corners
    else:
        for name, module1 in module._modules.items():
            module1 = recursion_change_bn1(module1)


def pth_to_jit(model, save_path, device="cuda:0", half=False):
    model.eval()
    input_x = torch.randn(1, 3, 112, 112).to(device)
    if half:
        input_x = input_x.half()
    new_model = torch.jit.trace(model, input_x)
    torch.jit.save(new_model, save_path)


def export_model_0_3_1(checkpoint_path, export_model_name, inputsize=[1, 3, 112, 112], combine_conv_bn=False, half=False):

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    pytorch_version_to_0_3_1()
    check_point = torch.load(checkpoint_path, map_location=device)
    model = init_backbone(name="MobileFaceNet", input_size=[112, 112])
    if combine_conv_bn:
        print("combine conv and bn...")
        fuse_module(model)
    state_dict = check_point['backbone']
    model = model.cuda() if device == f"cuda:0" else model
    mapped_state_dict = OrderedDict()
    for name, module in model._modules.items():
        recursion_change_bn1(module)
    for key, value in state_dict.items():
        # print(key)
        mapped_key = key
        mapped_state_dict[mapped_key] = value
        if 'num_batches_tracked' in key:
            del mapped_state_dict[key]
    model.load_state_dict(mapped_state_dict)
    if half:
        print("convert to fp16...")
        model = model.half()
    model.eval()
    # for key, value in model.state_dict().items():
    #     print(key)
    dummy_input = torch.autograd.Variable(torch.randn(inputsize))
    dummy_input = dummy_input.cuda() if device == f"cuda:0" else dummy_input
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)


def export_model(checkpoint_path, export_model_name, inputsize=[1, 3, 112, 112], combine_conv_bn=False, half=False):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    state_dict = check_point['backbone']
    model = init_backbone(name="MobileFaceNet", input_size=[112, 112])
    if combine_conv_bn:
        print("combine conv and bn...")
        fuse_module(model)
    model = model.to(device) if device == f"cuda:0" else model
    model.load_state_dict(state_dict)
    model.eval()
    # for key, value in state_dict.items():
    #     print(key)
    dummy_input = torch.randn(inputsize).to(device)
    if half:
        print("convert to fp16...")
        model = model.half()
        dummy_input = dummy_input.half()
    # torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True)
    torch.onnx.export(model=model, args=dummy_input, f=export_model_name, verbose=True, input_names=['image'],
                      output_names=['outTensor'], opset_version=11)  # 0.4.0以上支持更改输入输出层名称


def export_jit(checkpoint_path, export_model_name, combine_conv_bn=False, half=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    state_dict = check_point['backbone']
    model = init_backbone(name="MobileFaceNet", input_size=[112, 112])
    if combine_conv_bn:
        print("combine conv and bn...")
        fuse_module(model)
    model = model.cuda() if device == "cuda:0" else model
    model.load_state_dict(state_dict)
    if half:
        print("convert to fp16...")
        model = model.half()
    pth_to_jit(model, export_model_name, device, half=half)


def export_feature(checkpoint_path, export_model_name, combine_conv_bn=False, half=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    check_point = torch.load(checkpoint_path, map_location=device)
    state_dict = check_point['net']
    model = init_backbone(name="MobileFaceNet", input_size=[112, 112])
    if combine_conv_bn:
        print("combine conv and bn...")
        fuse_module(model)
    model = model.cuda() if device == "cuda:0" else model
    model.load_state_dict(state_dict)
    del model.classifier
    if half:
        print("convert to fp16...")
        model = model.half()
    print("export feature layers...")
    torch.save(model.state_dict(), export_model_name)


def main():
    if len(sys.argv) < 3:
        print("请提供需要转换的模型路径！")
        exit(-1)
    checkpoint_path = sys.argv[1]
    export_model_name = sys.argv[2]
    print("模型路径为：", os.path.realpath(checkpoint_path))
    transform_type = 0
    combine_conv_bn = False
    half = False
    if len(sys.argv) > 3:
        transform_type = int(sys.argv[3])
    if len(sys.argv) > 4:
        combine_conv_bn = int(sys.argv[4])
    if len(sys.argv) > 5:
        half = int(sys.argv[5])
    if transform_type == 0:
        export_feature(checkpoint_path, export_model_name, combine_conv_bn=combine_conv_bn, half=half)
    if transform_type == 1:
        if torch.__version__ >= "0.4.0":
            print(torch.__version__)
            print("pytorch version must <  0.3.1, please check it!")
            exit(-1)
        export_model_0_3_1(checkpoint_path=checkpoint_path, export_model_name=export_model_name,
                           combine_conv_bn=combine_conv_bn, half=half)
    if transform_type == 2:
        if torch.__version__ < "0.4.0":
            print("pytorch version must >  0.4.0, please check it!")
            exit(-1)
        export_model(checkpoint_path=checkpoint_path, export_model_name=export_model_name,
                     combine_conv_bn=combine_conv_bn, half=half)
    if transform_type == 3:
        if torch.__version__ < "1.0.0":
            print("pytorch version must >  1.0.0, please check it!")
            exit(-1)
        export_jit(checkpoint_path, export_model_name, combine_conv_bn=combine_conv_bn, half=half)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu_device) for gpu_device in [0]])
    main()

