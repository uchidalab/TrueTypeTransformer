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
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import os
import pickle
from tqdm import tqdm
import math


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
            root_dir: str,
            ref_file: str = 'google_font_category_v4.csv',
            filter: str = "large",
            label_request: str = 'character',
            request: str = "train",
            lim: int = 2838):
        super(QueryDataset, self).__init__()
        if not filter == "small" and not filter == "large":
            return print(f'your input is {filter}, expected "small" or "large"')
        self.root_dir = Path(root_dir)
        self.ref_file = ref_file
        self.filter = filter
        self.label_request = label_request
        self.request = request
        self.lim = lim
        self.font_names, self.font_df = self._get_font_paths()
        root = self.root_dir / '../Googlefonts'
        if not os.path.isfile(root / f'name_char_dict_{self.request}.pt'):
            self.ttfread(self.font_names, self.font_df)

        with open(root / f'{self.request}_dataset.pt', 'rb') as f:
            self.data = pickle.load(f)

        with open(root / f'name_char_dict_{self.request}.pt', 'rb') as f:
            self.name_char = pickle.load(f)
        # print(len(self.data), len(self.name_char))
        if self.lim < 500:
            self._lim_data(self.data, self.name_char)

        if self.label_request == 'character':
            self.label = [i[2] for i in self.name_char]

        elif self.label_request == 'style':
            self.label = [i[4] for i in self.name_char]
        # print(len(self.data), len(self.name_char))

    def __getitem__(self, index):
        return self.data[index], torch.tensor(self.label[index])

    def __len__(self):
        """Return numbr of font"""
        return len(self.label)
        # return 26*10

    def _lim_data(self, data, name_char):
        lim_data = []
        lim_name_char = []
        df = pd.DataFrame(name_char, columns=['font', 'char', 'char_id', 'style', 'style_id'])
        df['points'] = [len(i) for i in data]
        df_points = pd.DataFrame(df[df.points <= self.lim].groupby('font')['char'].count())
        NG_font = df_points[df_points.char != 26]['char'].index.tolist()

        for idx, lis in enumerate(name_char):
            if lis[0] in NG_font:
                continue
            lim_data.append(data[idx])
            lim_name_char.append(lis)

        self.data = lim_data
        self.name_char = lim_name_char
        return

    def _get_font_paths(self):
        """
        Get the list of available fonts and query information
        in the specified directory.
        """
        root_dir = self.root_dir
        df = pd.read_csv(self.ref_file, index_col=0)
        # num = df[(df.data_type == 'train') & (df.category != 'MONOSPACE') & (df.isin_latin)].shape[0]

        if self.request == 'train':
            train_fonts_ = []
            train_df = df[(df.data_type == 'train') & (df.category != 'MONOSPACE')
                          & (df.isin_latin)].copy().reset_index()
            for font_dir in train_df.font.tolist():
                for sub_dir in ['apache', 'ofl']:
                    train_fonts_.append(list((root_dir/sub_dir/font_dir).glob('*.ttf')))
            fonts = [j for i in train_fonts_ for j in i]
            df_fonts = train_df

        elif self.request == 'test':
            test_fonts_ = []
            test_df = df[(df.data_type == 'valid') & (df.category != 'MONOSPACE')
                         & (df.isin_latin)].copy().reset_index()
            for font_dir in test_df.font.tolist():
                for sub_dir in ['apache', 'ofl']:
                    test_fonts_.append(list((root_dir/sub_dir/font_dir).glob('*.ttf')))
            fonts = [j for i in test_fonts_ for j in i]
            df_fonts = test_df
        return fonts, df_fonts

    def ttfread(self, fonts, font_df):
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

        query = []
        font_error = []
        idx_name_char = []
        for font_url in tqdm(fonts):
            font = describe.openFont(font_url)
            for char in chars:
                g = glyph.Glyph(glyphquery.glyphName(font, char))
                try:
                    contours = g.calculateContours(font)
                    control_point = []
                    for contourID, contour in enumerate(contours):
                        orderID = 0
                        for (point_x, point_y), flag in contour:
                            control_point.append([point_x, point_y, flag, contourID, orderID])
                            orderID += 1
                    query.append(self.zero2one(torch.tensor(control_point), self.lim))
                    _char = char2label[char]
                    style = font_df[font_df['font'] == font_url.parent.name]['category'].tolist()[0]
                    _style = style2label[style]
                    idx_name_char.append([font_url.name,
                                          char,
                                          _char,
                                          style,
                                          _style])
                except Exception:
                    font_error.append(font_url.name)
                    break

        root = self.root_dir / '../Googlefonts'
        with open(root / f'name_char_dict_{self.request}.pt', 'wb') as f:
            pickle.dump(idx_name_char, f)
        with open(root / f'{self.request}_error_fontname.pt', 'wb') as f:
            pickle.dump(font_error, f)
        with open(root / f'{self.request}_dataset.pt', 'wb') as f:
            pickle.dump(query, f)
        print(f"all font {len(fonts) * 26} import font {len(query)} reject font {len(font_error)}")

        return

    def zero2one(self, data, lim):
        x = data[:, 0]
        y = data[:, 1]
        contourID = data[:, 3] / (lim-1)
        orderID = data[:, 4] / (lim-1)
        x = x + abs(x.min()) if x.min() < 0 else x - abs(x.min())
        y = y + abs(y.min()) if y.min() < 0 else y - abs(y.min())
        maxvalue = torch.stack([x, y]).max()
        return torch.stack([x/maxvalue, y/maxvalue, data[:, 2], contourID, orderID], axis=1)


if __name__ == "__main__":

    train_dataset = QueryDataset(request="train", label_request='style', lim=2838)
    # val_dataset = QueryDataset(request='val', label_request='style', lim=2838)
    test_dataset = QueryDataset(request='test', label_request='style', lim=2838)

    print()
    loader = DataLoader(train_dataset,
                        batch_size=256,
                        shuffle=False,
                        collate_fn=collate_fn)
    # for data, label in loader:
    #     print(data.shape, label.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    train_indexes = []
    val_indexes = []
    for train_index, val_index in skf.split(train_dataset[:][0], train_dataset[:][1]):
        train_indexes.append(train_index)
        val_indexes.append(val_index)
