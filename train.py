import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from collections import defaultdict


from utils import (
    sync_across_gpus,
    set_seed,
    get_model,
    create_checkpoint,
    get_data,
    get_dataset,
    get_dataloader,
    get_optimizer,
    get_scheduler,
)

from copy import copy
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import cv2

    cv2.setNumThreads(0)
except:
    print("no cv2 installed, running without")

sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")


def run_predict(model, test_dataloader, test_df, cfg, pre="test"):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = cfg.batch_to_device(data, cfg.device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]

        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.distributed and cfg.eval_ddp:
        for key, test in output.items():
            test_data[key] = sync_across_gpus(test_data[key], cfg.world_size)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            if cfg.distributed:
                for k, v in test_data.items():
                    test_data[k] = v[: len(test_dataloader.dataset)]
            torch.save(
                test_data,
                f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth",
            )

    if cfg.distributed:
        torch.distributed.barrier()

    print("TEST FINISHED")


def train(cfg):
    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    cfg.distributed = False
    if "WORLD_SIZE" in os.environ:
        cfg.distributed = int(os.environ["WORLD_SIZE"]) > 1

    if cfg.distributed:

        cfg.local_rank = int(os.environ["LOCAL_RANK"])

        print("RANK", cfg.local_rank)

        device = "cuda:%d" % cfg.local_rank
        cfg.device = device
        print("device", device)
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.world_size = torch.distributed.get_world_size()
        cfg.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process."
        )
        print(
            f"Process {cfg.rank}, total {cfg.world_size}, local rank {cfg.local_rank}."
        )
        cfg.group = torch.distributed.new_group(np.arange(cfg.world_size))
        print("Group", cfg.group)

        # syncing the random seed
        cfg.seed = int(
            sync_across_gpus(torch.Tensor([cfg.seed]).to(device), cfg.world_size)
            .detach()
            .cpu()
            .numpy()[0]
        )  #
        print("seed", cfg.local_rank, cfg.seed)

    else:
        cfg.local_rank = 0
        cfg.world_size = 1
        cfg.rank = 0

        device = "cuda:%d" % cfg.gpu
        cfg.device = device

    set_seed(cfg.seed)

    train_df, test_df = get_data(cfg)

    train_dataset = get_dataset(train_df, cfg, mode="train")
    train_dataloader = get_dataloader(train_dataset, cfg, mode="train")

    if cfg.test:
        test_dataset = get_dataset(test_df, cfg, mode="test")
        test_dataloader = get_dataloader(test_dataset, cfg, mode="test")
    model = get_model(cfg, train_dataset)
    model.to(device)

    if cfg.distributed:
        if cfg.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = NativeDDP(
            model, device_ids=[cfg.local_rank], find_unused_parameters=False
        )

    total_steps = len(train_dataset)
    if train_dataloader.sampler is not None:
        if "WeightedRandomSampler" in str(train_dataloader.sampler.__class__):
            total_steps = train_dataloader.sampler.num_samples

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    optimizer.zero_grad()
    val_score = 0
    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0:
            print("EPOCH:", epoch)

        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size * cfg.world_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")

                model.train()
                torch.set_grad_enabled(True)

                batch = cfg.batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if cfg.grad_accumulation != 0:
                    loss /= cfg.grad_accumulation

                # Backward pass

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:

                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                if cfg.distributed:
                    torch.cuda.synchronize()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

            print(f"Mean train_loss {np.mean(losses):.4f}")

        if cfg.distributed:
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(
                    cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
                )

                torch.save(
                    checkpoint,
                    f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth",
                )

    if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
        # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(
            cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
        )

        torch.save(
            checkpoint,
            f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth",
        )

    if cfg.test:
        run_predict(model, test_dataloader, test_df, cfg, pre="test")

    return val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-C", "--config", help="config filename")
    parser_args, other_args = parser.parse_known_args(sys.argv)
    cfg = copy(importlib.import_module(parser_args.config).cfg)

    # overwrite params in config with additional args
    if len(other_args) > 1:
        other_args = {
            k.replace("-", ""): v for k, v in zip(other_args[1::2], other_args[2::2])
        }

        for key in other_args:
            if key in cfg.__dict__:

                print(
                    f"overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}"
                )
                cfg_type = type(cfg.__dict__[key])
                if cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == "True"
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

    cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
    cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
    cfg.batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

    cfg.post_process_pipeline = importlib.import_module(
        cfg.post_process_pipeline
    ).post_process_pipeline

    result = train(cfg)
    print(result)
