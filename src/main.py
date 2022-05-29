import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from model.T3 import T3
from utils.load import get_loader
from utils.evaluate import ConfusionMatrix, EarlyStopping
from utils.train import train_model, eval_model
from pathlib import Path
import numpy as np
import shutil
import datetime
import glob
import hydra


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
            n_iter = train_model(model, train_loader, epoch, cfg.epoch,
                                 device, optimizer, writer, n_iter)

            loss = eval_model(model, val_loader, epoch, cfg.epoch, device, writer, n_iter)
            scheduler.step(loss/cfg.batch_size)
            # Check the earlystoppingの判定をさせる★
            earlystopping(loss, model, log_dir / f'epoch{epoch:05}.pt')
            save_ep = epoch
            save_model_dir = log_dir / f'epoch{save_ep:05}.pt'
            if earlystopping.early_stop:
                print('='*60)
                print('Early Stopping!')
                print('='*60)

                save_ep = epoch - cfg.patience
                save_model_dir = log_dir / f'epoch{save_ep:05}.pt'
                break
        print('Done.')
    except KeyboardInterrupt:
        print('='*60)
        print('Early Stopping!')
        print('='*60)
        save_ep = epoch - 1
        save_model_dir = log_dir / f'epoch{save_ep:05}.pt'
    if device == 'cuda':
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        print(elapsed_time / 1000, 'sec.')
        with open(log_dir / 'ElapsedTime.txt', 'w') as f:
            f.write(f'{elapsed_time / 1000} sec.')
    print('Make ConfusionMatrix and save the report')

    train_loader.shuffle = False
    phaze = ['train', 'val', 'test']
    for idx, loader in enumerate([train_loader, val_loader, test_loader]):
        name_char_dict_path = cwd / f'../data/Googlefonts/name_char_dict_{phaze[idx]}.pt'
        ConfusionMatrix(test_loader=loader,
                        model_select=model,
                        model_PATH=save_model_dir,
                        save_dir=log_dir,
                        save_name=f'{phaze[idx]}_{save_ep:05}',
                        device_select=device,
                        name_char_dict_path=name_char_dict_path,
                        save_dir_name=log_dir / f'miss_{phaze[idx]}_{save_ep:05}',
                        nb_classes=cfg.model.num_classes,
                        cwd=cwd,
                        error_img=False)


if __name__ == '__main__':
    main()
