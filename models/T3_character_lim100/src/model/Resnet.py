import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary


class ResNetFintune(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        channel: int = 3,
        num_classes: int = 26,
        dim_feedforward: int = 128
    ) -> None:
        super(ResNetFintune, self).__init__()
        self.input_shape = (channel, img_size, img_size)
        # class_num = 3
        self.net = models.resnet18(pretrained=False)
        # 出力層をnクラスに変更
        if channel == 1:
            self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x_):
        feat = self.net(x_)
        # print(feat.shape)
        feat = self.fc1(feat)
        y = self.fc2(feat)
        return y


if __name__ == '__main__':
    for num_classes in [26, 4]:
        for img_size in [16, 32, 64, 128, 256]:
            model = ResNetFintune(img_size=img_size, num_classes=num_classes)

            image_batch = torch.rand(256, *model.input_shape)
            print(image_batch.shape)
            image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
            model(image_batch)
            summary(model, (256, *model.input_shape), device='cpu')
