import torch
from torch import nn, Tensor
from torchinfo import summary


class CNNmodel1D(nn.Module):
    def __init__(
        self,
        *,
        font_dim: int = 100,
        word_size: int = 5,
        num_classes: int = 26,
        embed_dim: int = 100,
        dim_feedforward: int = 1024
    ) -> None:
        super(CNNmodel1D, self).__init__()

        self.input_shape = (font_dim, word_size)
        self.word_size = word_size
        self.embed_dim = embed_dim
        self.font_size = font_dim * embed_dim

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)

        self.conv1 = nn.Conv1d(word_size, 16, 3)
        self.conv2 = nn.Conv1d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 23, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)

    def forward(self, font: Tensor) -> Tensor:
        batch_size = len(font)
        x = font.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        *,
        font_dim: int = 100,
        word_size: int = 5,
        num_classes: int = 26,
        embed_dim: int = 100,
        dim_feedforward: int = 1024
    ) -> None:
        super(MLP, self).__init__()

        self.input_shape = (font_dim, word_size)
        self.word_size = word_size
        self.embed_dim = embed_dim
        self.font_size = font_dim * embed_dim

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)

        self.conv1 = nn.Conv1d(word_size, 16, 3)
        self.conv2 = nn.Conv1d(16, 32, 3)

        self.fc1 = nn.Linear(32 * 23, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)

    def forward(self, font: Tensor) -> Tensor:
        batch_size = len(font)
        x = font.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    font_dim = 100
    word_size = 5
    num_classes = 26

    model = CNNmodel1D(
        font_dim=font_dim,
        word_size=word_size,
        num_classes=num_classes)

    image_batch = torch.rand(256, *model.input_shape)
    print(image_batch.shape)
    image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
    model(image_batch)
    summary(model, (256, *model.input_shape), device='cpu')
