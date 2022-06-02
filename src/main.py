import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import datetime
import glob
import hydra

from model.T3 import T3
from utils.load import get_loader
from utils.evaluate import EarlyStopping
from utils.train import train_model, eval_model


today = datetime.date.today()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    # set initial value
    cwd = Path(hydra.utils.get_original_cwd())
    print(f'Orig working directory : {cwd}')

    # set device
    device = torch.device('cuda:' + str(cfg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(cfg.seed)
    # set logs dir
    num_id = len(glob.glob1(cwd / 'logs', f'{today:%Y%m%d}_*')) + 1
    writer = SummaryWriter(log_dir=cwd / f'logs/{today:%Y%m%d}_{num_id:02}_{cfg.method}')
    log_dir = Path(writer.get_logdir())

    shutil.copytree(cwd / 'src', log_dir / 'src')

    train_loader, val_loader, test_loader = get_loader(cfg, cwd)

    model = T3(
        font_dim=cfg.model.font_dim,
        word_size=cfg.model.word_size,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        heads=cfg.model.nhead,
        mlp_dim=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        emb_dropout=cfg.model.dropout,
        pool=cfg.model.pool
    )

    dumyinput = torch.rand(cfg.batch_size, *model.input_shape)
    dumyinput = torch.arange(dumyinput.numel()).reshape(dumyinput.shape).float()
    summary(model, (cfg.batch_size, *model.input_shape), device='cpu')
    # writer.add_graph(model, dumyinput)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=0.001)
    try:
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        n_iter = 0
        # Set earlyStopping
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
        for epoch in range(cfg.epoch):
            n_iter = train_model(model, train_loader, epoch, cfg.epoch, device, optimizer, writer, n_iter)

            loss = eval_model(model, val_loader, epoch, cfg.epoch, device, writer, n_iter)
            scheduler.step(loss/cfg.batch_size)
            # ★毎エポックearlystoppingの判定をさせる★
            earlystopping(loss, model, log_dir / f'epoch{epoch:05}.pt')  # callメソッド呼び出し
            if earlystopping.early_stop:  # ストップフラグがTrueの場合、breakでforループを抜ける
                print('='*60)
                print('Early Stopping!')
                print('='*60)
                break
        print('Done.')
    except KeyboardInterrupt:
        print('='*60)
        print('Early Stopping!')
        print('='*60)
    if device == 'cuda':
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        print(elapsed_time / 1000, 'sec.')
        with open(log_dir / 'ElapsedTime.txt', 'w') as f:
            f.write(f'{elapsed_time / 1000} sec.')

    # test accuracy
    model.eval()

    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.max(1)[1] == labels).sum().item()
            total += len(labels)
        loss /= total
        accuracy = correct / total
    print(f'test loss : {loss} test accuracy : {accuracy}')


if __name__ == '__main__':
    main()
