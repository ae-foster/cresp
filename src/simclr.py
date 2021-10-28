"""Train an encoder using Contrastive Learning."""
import argparse
import os
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from optimizers import LARS  # from torchlars import LARS

from tqdm import tqdm

from dataset import get_datasets
from critic import LinearCriticSimCLR as LinearCritic
from evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(args):
    # Check whether config already exists and job has been successful
    run_path = os.getcwd()
    success_path, checkpoint_path = manage_exisiting_config(run_path)

    args.lr = args.base_lr * (args.batch_size / 256)
    with open(".hydra/config.yaml", "r") as fp:
        OmegaConf.save(config=args, f=fp.name)

    # For reproducibility purposes
    args.seed = args.run if args.seed == 0 else int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    pl.seed_everything(args.seed)
    save_reproduce(sys.argv, args.seed, run_path, git_hash)

    log.info("Run in parallel with {} gpus".format(torch.cuda.device_count()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    clf = None

    log.info("==> Preparing data..")
    trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    log.info("==> Building model..")
    ##############################################################
    # Encoder
    ##############################################################
    if args.arch in ["ResNet18", "ResNet34", "ResNet50"]:
        net = eval(args.arch)(stem=stem, num_channels=args.dataset.shape[0]).to(device)
        args.representation_dim = args.representation_dim
    else:
        raise ValueError("Bad architecture specification")

    ##############################################################
    # Critic
    ##############################################################
    critic = LinearCritic(net.representation_dim, temperature=args.temperature).to(device)

    if device == "cuda":
        repr_dim = net.representation_dim
        net = torch.nn.DataParallel(net)
        net.representation_dim = repr_dim

    if checkpoint_path is not None:
        # Load checkpoint.
        log.info("==> Resuming from checkpoint..")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["net"])
        critic.load_state_dict(checkpoint["critic"])
        best_acc = checkpoint["acc"] if "acc" in checkpoint else 0.0
        start_epoch = checkpoint["epoch"]

    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-6, momentum=args.momentum)
    if args.cosine_anneal:
        scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
    encoder_optimizer = LARS(base_optimizer, 1e-3)

    # Training
    def train(epoch):
        log.info("\nEpoch: %d" % epoch)
        net.train()
        critic.train()
        train_loss = 0
        t = tqdm(enumerate(trainloader), desc="Loss: **** ", total=len(trainloader), bar_format="{desc}{bar}{r_bar}")
        for batch_idx, (inputs, _, _) in t:
            x1, x2 = inputs
            x1, x2 = x1.to(device), x2.to(device)
            encoder_optimizer.zero_grad()
            representation1, representation2 = net(x1), net(x2)
            raw_scores, pseudotargets = critic(representation1, representation2)
            loss = criterion(raw_scores, pseudotargets)
            loss.backward()
            encoder_optimizer.step()

            train_loss += loss.item()

            t.set_description("Loss: %.3f " % (train_loss / (batch_idx + 1)))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch)
        if (args.val_freq > 0) and (epoch % args.val_freq == (args.val_freq - 1)):
            X, y = encode_train_set(clftrainloader, device, net)
            clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
            acc = test(testloader, device, net, clf)
            if acc > best_acc:
                best_acc = acc
            save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
        elif args.val_freq == 0:
            save_checkpoint(net, clf, critic, epoch, args, os.path.basename(__file__))
        if args.cosine_anneal:
            scheduler.step()

    open(success_path, "w+").close()


if __name__ == "__main__":
    cudnn.benchmark = True
    # create directory in `scratch` drive and symlink experiments to it
    # create_symlink('conf/config.yaml')
    git_hash = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"])

    main()