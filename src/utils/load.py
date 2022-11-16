###############
# We use Googlefonts
# https://github.com/google/fonts.git
# And 'google_font_category_v4.csv'

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ttfquery import describe
from ttfquery import glyphquery
import ttfquery.glyph as glyph
import pandas as pd
from pathlib import Path
from os.path import basename, dirname, exists
import pickle
from tqdm import tqdm


def get_loader(cfg, cwd):
    train_dataset = QueryDataset(root_dir=cwd / cfg.root, ref_file=cwd / cfg.ref_file,
                                 label_request=cfg.label_request, split='train', filter='large')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=2)

    validation_dataset = QueryDataset(root_dir=cwd / cfg.root, ref_file=cwd / cfg.ref_file,
                                      label_request=cfg.label_request, split='val', filter='large')

    val_loader = DataLoader(validation_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=2)

    test_dataset = QueryDataset(root_dir=cwd / cfg.root, ref_file=cwd / cfg.ref_file,
                                label_request=cfg.label_request, split='test', filter='large')

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=2)
    return train_loader, val_loader, test_loader


def collate_fn(data):
    datas, labels = zip(*data)
    x = pad_sequence(datas, batch_first=True, padding_value=-1)
    return x, torch.tensor(labels)


class QueryDataset(Dataset):
    def __init__(
            self,
            root_dir: str = '../data/fonts',
            root_sub: str = 'Ours_t3',
            ref_file: str = 'google_font_category_v4.csv',
            meta_file: str = 'google_font_metafile.csv',
            label_request: str = 'character',
            split: str = "train",
            char_select: str = '*',
            target_char: str = '*',
            filter: str = 'large',
            target: bool = False
    ):
        super(QueryDataset, self).__init__()
        if not filter == "small" and not filter == "large":
            return print(f'your input is {filter}, expected "small" or "large"')
        self.root_dir = Path(root_dir)
        self.rootsub = root_sub
        self.ref_file = ref_file
        self.filter = filter
        self.label_request = label_request
        self.split = split
        self.font_df = None
        self.target = target
        # 自身で作成したメタファイルを読み込み
        self.meta_df = pd.read_csv(meta_file, index_col=0)
        self.font_names = self._get_font_paths()

        if self.split != 'val':
            if not exists(self.root_dir / self.rootsub / f'name_char_dict_{self.split}.pt'):
                self.ttfread(self.font_names, self.font_df)
            assert exists(self.root_dir / self.rootsub / f'{self.split}_dataset'), 'Not exist dataset'
            # split; train or test
            data_dir = self.root_dir / self.rootsub / f'{self.split}_dataset'
        else:
            data_dir = self.root_dir / self.rootsub / 'train_dataset'

        self.input_data = self._get_datapath(data_dir, char_select)
        self.target_data = self._get_datapath(data_dir, target_char)

        print('N of input data', len(self.input_data), ', N of target data', len(self.target_data))

        num = [i for i in range(26)]
        chars = [chr(i) for i in range(65, 65+26)]
        char2label = dict(zip(chars, num))

        num_style = [i for i in range(4)]
        styles = ['SANS_SERIF', 'HANDWRITING', 'DISPLAY', 'SERIF']
        style2label = dict(zip(styles, num_style))
        if self.label_request == 'character':
            self.input_label = [char2label[basename(_)[:-3].split('-')[1]] for _ in self.input_data]
            self.target_label = [char2label[basename(_)[:-3].split('-')[1]] for _ in self.target_data]
        if self.label_request == 'style':
            self.input_label = [style2label[basename(_)[:-3].split('-')[2]] for _ in self.input_data]
            self.target_label = [style2label[basename(_)[:-3].split('-')[2]] for _ in self.target_data]

    def __getitem__(self, index):
        input_data = torch.load(self.input_data[index])
        target_data = torch.load(self.target_data[index])
        input_label = self.input_label[index]
        target_label = self.target_label[index]
        if self.target:
            return input_data, target_data, input_label, target_label
        return input_data, input_label

    def __len__(self):
        """Return numbr of font"""
        return len(self.input_data)
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
        if self.split == 'train':
            return train_path

        if self.split == 'test':
            return test_path

    def _get_datapath(self, data_dir, char_select):
        # 選択した文字のパスだけを抽出

        def paths_sorted(paths):
            return sorted(paths, key=lambda x: str(x.name))

        sorted_data = paths_sorted(list(data_dir.glob(f'*-{char_select}-*.pt')))

        _df = self.meta_df[(self.meta_df['split'] == self.split)].groupby('font_name').count()
        idx = self.meta_df[self.meta_df['font_name'].isin(_df[_df['index'] == 26].index.tolist())]['index'].tolist()

        idx = list(map(lambda k: f'{k:05}', idx))
        return [x for x in sorted_data if basename(x)[:-3].split('-')[0] in idx]

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
        (root / f'{self.split}_dataset').mkdir(parents=True, exist_ok=True)

        idx = 0
        if self.split == 'test':
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
                torch.save(self.zero2one(torch.tensor(control_point)),  root / f'{self.split}_dataset/{idx:05}-{char}-{style}.pt')
                idx += 1
                # save meta data about fontname, character, font style
                _char = char2label[char]
                _style = style2label[style]
                idx_name_char.append([font_url.name,
                                      char,
                                      _char,
                                      style,
                                      _style])

        with open(root / f'name_char_dict_{self.split}.pt', 'wb') as f:
            pickle.dump(idx_name_char, f)
        print(f"all font {len(font_urls) * 26} import font {idx}")

        return

    def zero2one(self, data):
        x = data[:, 0]
        y = data[:, 1]
        contourID = data[:, 3]
        orderID = data[:, 4]
        x = x + abs(x.min()) if x.min() < 0 else x - abs(x.min())
        y = y + abs(y.min()) if y.min() < 0 else y - abs(y.min())
        maxvalue = torch.stack([x, y]).max()
        return torch.stack([x/maxvalue, y/maxvalue, data[:, 2], contourID, orderID], axis=1)


if __name__ == "__main__":

    train_dataset = QueryDataset(request="train", label_request='style', lim=2838)
    test_dataset = QueryDataset(request='test', label_request='style', lim=2838)
