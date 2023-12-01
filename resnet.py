import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb


def get_resnet(model_name, dataset="cifar-10"):

    if model_name == "resnet18":
        model = models.resnet18()
    elif model_name == "resnet34":
        model = models.resnet34()
    elif model_name == "resnet50":
        model = models.resnet50()
    else:
        raise ValueError("model_name not supported")

    if dataset == "cifar-10":
        num_classes = 10
    elif dataset == "cifar-100":
        num_classes = 100
    elif dataset == "clothing1M":
        num_classes = 14
    else:
        raise ValueError("dataset not supported")

    if dataset in ["cifar-10", "cifar-100"]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    net = get_resnet("resnet18", "cifar-10")
    batch = torch.randn(8, 3, 32, 32)
    output = net(batch)
    print(output.shape)
    ipdb.set_trace()
    pass

