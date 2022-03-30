import torch
from torch import nn, Tensor
from torchinfo import summary
import math


class CNNmodel2D(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 32,
        channel: int = 3,
        num_classes: int = 26,
        dim_feedforward: int = 1024
    ) -> None:
        super(CNNmodel2D, self).__init__()
        self.input_shape = (channel, img_size, img_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

        self.fc1 = nn.Linear(16 * 8 ** 2, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)

        depth = int(math.log2(img_size) - 4)
        self.conv_layers = nn.ModuleList([])
        for _ in range(depth):
            self.conv_layers.append(nn.ModuleList([
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            ]))

    def forward(self, img: Tensor) -> Tensor:
        batch_size = img.shape[0]
        x = self.conv1(img)
        x = self.relu(x)
        x = self.pool(x)
        for conv, relu, pool in self.conv_layers:
            x = conv(x)
            x = relu(x)
            x = pool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    font_dim = 100
    word_size = 5
    num_classes = 26
    for img_size in [256, 128, 64, 32, 16]:
        model = CNNmodel2D(img_size=img_size)

        image_batch = torch.rand(256, *model.input_shape)
        print(image_batch.shape)
        image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
        model(image_batch)
        summary(model, (256, *model.input_shape), device='cpu')
