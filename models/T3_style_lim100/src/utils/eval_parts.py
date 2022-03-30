import numpy as np
import torch
from load import QueryDataset
from torch.utils.data import DataLoader
import pickle
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
# from model.Transformer import TransformerModel
from sklearn.decomposition import PCA


def font_cut_out():
    im = Image.open(f'./font/bitmap/{name[:-4]}/{char}.png')
    img = np.asarray(im)
    img2 = np.where(img == 255, 0, 1)
    plt.imshow(img2)
    plt.show()
    np.unique(img2)
    hoge = np.where(img2 == 1)
    wMax, wMin, hMax, hMin = hoge[0].max(), hoge[0].min(), hoge[1].max(), hoge[1].min()
    plt.imshow(img[wMin:wMax+1, hMin:hMax+1]), plt.show()


def outline_plot():
    for idx in range(80, 6556, 456):
        name = fontname[idx]
        char = 'Q'
        data = np.array(datadict[name][char])
        data_norm = zero2one(data)
        fig = plt.figure(figsize=figsize)
        cm = plt.cm.get_cmap('RdYlBu')
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1)

        im = Image.open(f'/home/yusuke/Dev/data/font/bitmap/{name[:-4]}/{char}.png')
        img = np.asarray(im)
        img2 = np.where(img == 255, 0, 1)
        hoge = np.where(img2 == 1)

        hMax, hMin, wMax, wMin = hoge[0].max(), hoge[0].min(), hoge[1].max(), hoge[1].min()
        ax.imshow(img[hMin:hMax+1, wMin:wMax+1], alpha=0.6)
        width = False
        hight = False
        if (hMax - hMin) < (wMax-wMin):
            res = wMax-wMin
            width = True
        elif (hMax - hMin) > (wMax-wMin):
            res = hMax - hMin
            hight = True
        x = []
        y = []
        f = []
        if width:
            for i in range(data_norm.shape[0]):
                x.append(data_norm[i, 0] * res)
                y.append(data_norm[i, 1] * res)
                f.append(data_norm[i, 2])

        elif hight:
            for i in range(data_norm.shape[0]):
                x.append(data_norm[i, 0] * res)
                y.append(res-(data_norm[i, 1] * res))
                f.append(data_norm[i, 2])
        im = ax.scatter(x, y, s=100, c=f, linewidths=0, alpha=1, cmap=cm)
        fig.colorbar(im, ax=ax)
        plt.title(idx)
        plt.show()


def scat(data, name, figsize):
    fig = plt.figure(figsize=figsize)
    cm = plt.cm.get_cmap('RdYlBu')
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1)
    # back_im = Image.open(f'./font/bitmap/{name[:-4]}/{char}.png')
    # ax.imshow(im, alpha=0.6)
    im = ax.scatter(data[:, 0], data[:, 1], s=50, c=data[:, 2], linewidths=0, alpha=1, cmap=cm)
    fig.colorbar(im, ax=ax)
    plt.title(name)
    plt.show()


def zero2one(data, lim=100):
    x = data[:, 0]
    y = data[:, 1]
    contourID = data[:, 3]/(lim-1)
    orderID = data[:, 4]/(lim-1)
    x = x + abs(x.min()) if x.min() < 0 else x - abs(x.min())
    y = y + abs(y.min()) if y.min() < 0 else y - abs(y.min())
    maxvalue = np.stack([x, y]).max()
    return np.stack([x/maxvalue, y/maxvalue, data[:, 2], contourID, orderID], axis=1)


def extract(target, inputs):
    features = None

    def forward_hook(module, inputs, outputs):
        # 順伝搬の出力を features というグローバル変数に記録する
        global features
        features = outputs.detach()

    # コールバック関数を登録する。
    handle = target.register_forward_hook(forward_hook)

    # 推論する
    model.eval()
    model(inputs)

    # コールバック関数を解除する。
    handle.remove()

    return features


if __name__ == '__main__':

    PATH = './runs/Jul06_09-04-15_yusuke-lab/epoch04400.pt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    '''
    model = TransformerModel(font_dim=100, word_size=5, num_classes=26)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    print(model)

    target_module = model.pos_embedding
    print(target_module)'''

    sns.set(style="darkgrid", palette="muted", color_codes=True)
    with open('../data/font/large_dataset.pt', 'rb') as f:
        datadict = pickle.load(f)
    with open('../data/font/large/query_lim100_val.pt', 'rb') as f:
        query = pickle.load(f)
    fontname = list(datadict.keys()).copy()
    chars = [chr(i) for i in range(65, 65+26)]
    name = next(iter(fontname))
    char = next(iter(chars))
    data = np.array(datadict[name][char])
    figsize = (data[:, 0].max()-data[:, 0].min())/100, (data[:, 1].max()-data[:, 1].min())/100

    '''main'''
    validation_dataset = QueryDataset(font_dir="../data/ultimatefont/",
                                      root_dir="../data/font/", filter="large", request='val')
    val_loader = DataLoader(validation_dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=2)

    for data, label in val_loader:
        features = extract(target_module, data)
        print(data.shape, features.shape)
        break

    fdata = data[0].to('cpu').detach().numpy().copy()
    figsize = (fdata[:, 0].max()*10, fdata[:, 1].max()*10)
    scat(fdata, 'before embed', figsize)

    fea = features[0].to('cpu').detach().numpy().copy()

    pca = PCA(n_components=5)
    pca.fit(np.transpose(fea))
    print(pca.components_.shape)
    fead = pca.transform(np.transpose(fea))
    scat(fead, 'after embed', figsize)

    fdata = data.to('cpu').detach().numpy().copy()
    print(np.linalg.matrix_rank(fdata))
    f = features.to('cpu').detach().numpy().copy()
    print(np.linalg.matrix_rank(f), features.shape)
