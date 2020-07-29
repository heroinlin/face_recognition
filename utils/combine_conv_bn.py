import torch
import torch.nn as nn
import torchvision as tv


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        groups=conv.groups,
        bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and c is not None:
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = nn.Identity()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            c = None
            fuse_module(child)


def main():
    dummy_input = torch.randn([1, 3, 224, 224])
    model = tv.models.resnet18()
    model.eval()
    with torch.no_grad():
        origin_output = model(dummy_input)
    fuse_module(model)
    with torch.no_grad():
        new_output = model(dummy_input)
        bias = torch.abs(new_output - origin_output)
    print("max bias:", bias.max())
    torch.onnx.export(model, (dummy_input,), "opt_resnet18.onnx")


if __name__ == "__main__":
    main()
