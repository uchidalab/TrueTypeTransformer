###############
# We use Googlefonts
# https://github.com/google/fonts.git
# And 'google_font_category_v4.csv'

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import pandas as pd
from pathlib import Path
from os.path import basename, dirname, exists
import numpy as np
import pickle
from tqdm import tqdm
import math
from typing import Optional


def get_loader(cfg, cwd):

    dataset = QueryDataset(root_dir=cwd / cfg.root_dir, ref_file=cfg.ref_file,
                           filter=cfg.filter, label_request=cfg.label_request,
                           request='train', lim=cfg.lim)
    train_dataset = Subset(dataset, [i for i in range(math.ceil(len(dataset)/26*.9) * 26)])
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=2)

    validation_dataset = Subset(dataset, [i for i in range(math.ceil(len(dataset)/26*.9) * 26, len(dataset))])
    val_loader = DataLoader(validation_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=2)

    test_dataset = QueryDataset(root_dir=cwd / cfg.root_dir, ref_file=cfg.ref_file,
                                filter=cfg.filter, label_request=cfg.label_request,
                                request='test', lim=cfg.lim)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=2)
    return train_loader, val_loader, test_loader


def collate_fn(data):
    datas, labels = zip(*data)
    x = torch.nn.utils.rnn.pad_sequence(datas, batch_first=True, padding_value=-1)
    return x, torch.tensor(labels)


class QueryDataset(Dataset):
    def __init__(
            self,
            root_dir: str = Path('../data/fonts'),
            root_sub='./OursDataset_id_255',
            ref_file: str = 'google_font_category_v4.csv',
            label_request: str = 'character',
            request: str = "train",
            lim: int = 2838,
            char_select: str = '*',
            filter='large'):
        super(QueryDataset, self).__init__()
        if not filter == "small" and not filter == "large":
            return print(f'your input is {filter}, expected "small" or "large"')
        self.root_dir = Path(root_dir)
        self.ref_file = ref_file
        self.filter = filter
        self.label_request = label_request
        self.request = request
        self.lim = lim
        self.font_df = None
        self.font_names = self._get_font_paths()
        self.rootsub = root_sub

        if not exists(self.root_dir / self.rootsub / f'name_char_dict_{self.request}.pt'):
            self.ttfread(self.font_names, self.font_df)
        assert exists(self.root_dir / self.rootsub / f'{self.request}_dataset'), 'Not exist dataset'
        data_dir = self.root_dir / self.rootsub / f'{self.request}_dataset'

        def paths_sorted(paths):
            return sorted(paths, key=lambda x: str(x.name))

        self.data = [p for p in paths_sorted(list(data_dir.glob(f'*-{char_select}-*.pt')))]

        num = [i for i in range(26)]
        chars = [chr(i) for i in range(65, 65+26)]
        char2label = dict(zip(chars, num))

        num_style = [i for i in range(4)]
        styles = ['SANS_SERIF', 'HANDWRITING', 'DISPLAY', 'SERIF']
        style2label = dict(zip(styles, num_style))

        if self.label_request == 'character':
            self.label = [char2label[basename(_)[:-3].split('-')[1]] for _ in self.data]
        elif self.label_request == 'style':
            self.label = [style2label[basename(_)[:-3].split('-')[2]] for _ in self.data]

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        if self.label_request == 'character':
            label = self.label[index]
        elif self.label_request == 'style':
            label = self.label[index]
        return data, label

    def __len__(self):
        """Return numbr of font"""
        return len(self.data)
        # return 26*10

    def _get_font_paths(self):
        """
        Get the list of available fonts and query information
        in the specified directory.
        """
        self.font_df = pd.read_csv(self.ref_file, index_col=0)
        meta_df = self.font_df
        train_list = meta_df[(meta_df['data_type'] == 'train') &
                             (meta_df.category != 'MONOSPACE') & (meta_df.isin_latin)]['font'].tolist()
        test_list = meta_df[(meta_df['data_type'] == 'valid') &
                            (meta_df.category != 'MONOSPACE') & (meta_df.isin_latin)]['font'].tolist()

        all_path = list(self.root_dir.glob('**/*.ttf'))
        train_path = [x for x in all_path if basename(dirname(x)) in train_list]
        test_path = [x for x in all_path if basename(dirname(x)) in test_list]
        if self.request == 'train':
            return train_path

        if self.request == 'test':
            return test_path

    def ttfread(self, font_urls, font_df):
        chars = []
        if self.filter == "small":
            chars = [chr(i) for i in range(97, 97+26)]
        elif self.filter == "large":
            chars = [chr(i) for i in range(65, 65+26)]
        num = [i for i in range(26)]
        char2label = dict(zip(chars, num))

        num_style = [i for i in range(4)]
        styles = ['SANS_SERIF', 'HANDWRITING', 'DISPLAY', 'SERIF']
        style2label = dict(zip(styles, num_style))

        idx_name_char = []
        root = self.root_dir / self.rootsub
        root.mkdir(exist_ok=True)
        (root / f'{self.request}_dataset').mkdir(parents=True, exist_ok=True)

        idx = 0
        if self.request == 'test':
            idx = 1353 * 26  # train fonts: 1353 * characters: 26
        for font_url in tqdm(font_urls):
            font = describe.openFont(font_url)
            for char in chars:
                g = glyph.Glyph(glyphquery.glyphName(font, char))
                contours = g.calculateContours(font)
                control_point = []
                for contourID, contour in enumerate(contours):
                    orderID = 0
                    for (point_x, point_y), flag in contour:
                        control_point.append([point_x, point_y, flag, contourID, orderID])
                        orderID += 1
                style = font_df[font_df['font'] == font_url.parent.name]['category'].tolist()[0]
                x_min, y_min, _, _, _ = np.array(control_point).min(axis=0)
                x_max, y_max, _, _, _ = np.array(control_point).max(axis=0)
                torch.save(self.zero2max(torch.tensor(control_point), x_min, x_max, y_min, y_max),
                           root / f'{self.request}_dataset/{idx:05}-{char}-{style}.pt')
                # torch.save(torch.tensor(control_point),  root / f'{self.request}_dataset/{idx:05}-{char}-{style}.pt')
                idx += 1
                # save meta data about fontname, character, font style
                _char = char2label[char]
                _style = style2label[style]
                idx_name_char.append([font_url.name,
                                      char,
                                      _char,
                                      style,
                                      _style])

        with open(root / f'name_char_dict_{self.request}.pt', 'wb') as f:
            pickle.dump(idx_name_char, f)
        print(f"all font {len(font_urls) * 26} import font {idx}")

        return

    def zero2one(self, data, x_min, x_max, y_min, y_max, zero2max=False) -> Optional[list]:
        maxvalue = max([x_max - x_min, y_max - y_min])
        x, y = data[:, 0], data[:, 1]
        x = x + abs(x_min) if x_min < 0 else x - abs(x_min)
        y = y + abs(y_min) if y_min < 0 else y - abs(y_min)
        if zero2max:
            return torch.stack([torch.trunc(x/maxvalue * 255), torch.trunc(y/maxvalue * 255),
                                data[:, 2], data[:, 3], data[:, 4]], axis=1)
        return torch.stack([x/maxvalue, y/maxvalue, data[:, 2], data[:, 3], data[:, 4]], axis=1)

    def zero2max(self, data, x_min, x_max, y_min, y_max) -> Optional[list]:
        return self.zero2one(data, x_min, x_max, y_min, y_max, zero2max=True)

    def add_sos_eos(self, data):
        P, S, args = data.shape
        # input.shape = (P, Seq_len, args_len), output.shape = (P, Seq_len + 2, args_len)
        return torch.cat((torch.full((P, 1, args), self.pad_value), data, torch.full((P, 1, args), self.pad_value)), dim=1)

    def add_sos_eos_CMD(self, data):
        P, S = data.shape
        # 'SOS' = 2, 'EOS' = 3, output.shape = (P, Seq_len), output.shape = (P, Seq_len + 1)
        return torch.cat((torch.full((P, 1), self.sos_value), data, torch.full((P, 1), self.eos_value)), dim=1)


if __name__ == "__main__":

    train_dataset = QueryDataset(request="train", label_request='style', lim=2838)
    test_dataset = QueryDataset(request='test', label_request='style', lim=2838)
