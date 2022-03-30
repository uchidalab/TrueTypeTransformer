import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from pathlib import Path
from PIL import Image
import pickle


def get_loader(cfg, cwd):
    dataset = ImgDataset(
        root=cwd / f'{cfg.model.data_path}/{cfg.model.img_size}',
        transform=torchvision.transforms.ToTensor(),
        filter=cfg.filter,
        label_request=cfg.label_request,
        request='train'
    )

    train_dataset = Subset(dataset, [i for i in range(519 * 26)])

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=2)

    validation_dataset = Subset(dataset, [i for i in range(519 * 26, 576 * 26)])
    val_loader = DataLoader(validation_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=2)

    test_dataset = ImgDataset(
        root=cwd / f'{cfg.model.data_path}/{cfg.model.img_size}',
        transform=torchvision.transforms.ToTensor(),
        filter=cfg.filter,
        label_request=cfg.label_request,
        request='test'
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=2)
    return train_loader, val_loader, test_loader


class ImgDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            filter: str = "large",
            label_request: str = 'character',
            request: str = "train") -> None:
        super().__init__()
        self.transform = transform
        self.request = request
        self.filter = filter
        self.label_request = label_request
        self.data = []
        self.labels = []
        self.name = []
        self.root_dir = Path(root)
        with open(root / f'../../name_char_dict_{self.request}.pt', 'rb') as f:
            name_char = pickle.load(f)
        # ['CraftyGirls-Regular.ttf', 'A', 0, 'HANDWRITING', 1]
        for names in name_char:
            font, char, char_id, style, style_id = names
            self.data.append(root / font[:-4] / f'{char}.png')
            if self.label_request == 'character':
                self.labels.append(char_id)
            elif self.label_request == 'style':
                self.labels.append(style_id)

    def __getitem__(self, index):
        img_path = self.data[index]
        label = torch.tensor(self.labels[index])

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    data_path = Path('../data/font/Ultimate_img_32')
    train_dataset = ImgDataset(
        root=data_path / 'train',
        transform=torchvision.transforms.ToTensor()
    )
    print(len(train_dataset))
    val_dataset = ImgDataset(
        root=data_path / 'val',
        transform=torchvision.transforms.ToTensor()
    )
    print(len(val_dataset))
    test_dataset = ImgDataset(
        root=data_path / 'test',
        transform=torchvision.transforms.ToTensor()
    )
    print(len(test_dataset))
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        num_workers=2,
        shuffle=False
    )
