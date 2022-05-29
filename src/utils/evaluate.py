import torch
from sklearn.metrics import classification_report
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from src.utils.evaluate import ConfusionMatrix

sns.set(style="darkgrid", palette="muted", color_codes=True)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience 
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = ''

    def __call__(self, val_loss, model, path):
        score = -val_loss
        self.path = path

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)

        # elif score < self.best_score:
        #     self.counter += 1
        #     if self.verbose:
        #         print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model :{self.path}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def ConfusionMatrix(test_loader, model_select, model_PATH, save_dir, save_name, device_select,
                    name_char_dict_path, save_dir_name, nb_classes, cwd, error_img=False):
    device = device_select
    model = model_select
    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    model.to(device)
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    num = [i for i in range(26)]
    chars = [chr(i) for i in range(65, 65+26)]
    num_style = [i for i in range(4)]
    styles = ['SANS_SERIF', 'HANDWRITING', 'DISPLAY', 'SERIF']
    label2class = dict(zip(num, chars))
    label2style = dict(zip(num_style, styles))
    trueclasses = []
    predclasses = []
    inputsdata = []
    with torch.no_grad():
        for inputs, classes in tqdm(test_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            inputsdata.append(inputs.to(device='cpu').detach().numpy().copy())
            trueclasses.append(classes.to(device='cpu').detach().numpy().copy())
            predclasses.append(preds.to(device='cpu').detach().numpy().copy())
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    y_true = [j for i in trueclasses for j in i.tolist()]
    y_pred = [j for i in predclasses for j in i.tolist()]

    plt.figure(figsize=(15, 10))
    if outputs.shape[1] == 4:
        class_names = list(label2style.values())
    else:
        class_names = list(label2class.values())
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'{save_name}')
    plt.savefig(f'{save_dir}/{save_name}.jpg')
    report = pd.DataFrame(classification_report(y_true, y_pred, target_names=class_names, output_dict=True))
    print(report)
    report.T.to_csv(f'{save_dir}/{save_name}.csv')
    # plt.show()
    plt.clf()
    plt.close()

    if error_img:
        X = np.array([j for i in inputsdata for j in i.tolist()])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ill = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

        print(f'{len(ill)}/{len(X)}, acc:{1 - (float(len(ill))/float(len(X)))}')

        with open(name_char_dict_path, 'rb') as f:
            name_char = pickle.load(f)
        for idx, data in tqdm(enumerate(X[ill])):
            name = np.array(name_char)[ill][idx].tolist()[0]
            char = np.array(name_char)[ill][idx].tolist()[1]
            if outputs.shape[1] == 4:
                save_name_img = \
                    f'{idx: 05}_Pred: {label2style[y_pred[ill][idx]]} True: {label2style[y_true[ill][idx]]}_{name[:-4]}'
            else:
                save_name_img = \
                    f'{idx: 05}_Pred: {label2class[y_pred[ill][idx]]} True: {label2class[y_true[ill][idx]]}_{name[:-4]}'
            data = np.array(data)
            plotdata = data[:np.where(data == -1)[0][0], :] if len(np.where(data == -1)[0]) != 0 else data

            plot_outline_on_img(plotdata, name, char, save_name_img, save_dir_name, cwd)
    return df_cm


def plot_outline(data, name, save_dir):
    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']
    cmap = ListedColormap(colors, name="custom")
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.scatter(data[:, 0], data[:, 1], s=50, c=data[:, 2], linewidths=0, alpha=1, cmap=cmap)
    fig.colorbar(im, ax=ax)
    plt.title(name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{name}.jpg'))
    # plt.show()
    plt.clf()
    plt.close()


def plot_outline_on_img(plotdata, name, char, save_name, save_dir_name, cwd):
    fig = plt.figure(figsize=(10, 10))
    colors = ['red', 'blue']
    cmap = ListedColormap(colors, name="custom")
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1)
    try:
        im = Image.open(cwd / f'../data/Googlefonts/img/256/{name[:-4]}/{char}.png')
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
        if width:
            for i in range(plotdata.shape[0]):
                x.append(plotdata[i, 0] * res)
                y.append((hMax - hMin) - plotdata[i, 1] * res)

        elif hight:
            for i in range(plotdata.shape[0]):
                x.append(plotdata[i, 0] * res)
                y.append(res-(plotdata[i, 1] * res))

        else:
            res = hMax - hMin
            for i in range(plotdata.shape[0]):
                x.append(plotdata[i, 0] * res)
                y.append(res-(plotdata[i, 1] * res))

        im = ax.scatter(x, y, s=100, c=plotdata[:, 2], linewidths=0, alpha=1, cmap=cmap)
        fig.colorbar(im)
        ax.set_xticks([])
        ax.set_xlim(min(x) - 10, max(x) + 10)
        ax.set_yticks([])
        ax.set_ylim(min(y) - 10, max(y) + 10)
        ax.invert_yaxis()
        plt.title(save_name, size=24)
        os.makedirs(save_dir_name, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_dir_name / f'{save_name}.jpg')
        # plt.show()
        plt.clf()
        plt.close()
    except FileNotFoundError:
        plot_outline(plotdata, save_name, save_dir_name)
