import argparse
import yaml
import torch
import sys
import wandb
import numpy as np
import warnings
import soundfile as sf
import time
import os
import gc
import io
import matplotlib.pyplot as plt

from PIL import Image
from typing import Union, List, Callable, Dict, Any, Optional
from utils import get_model_from_config, get_optimizer, normalize_batch, get_scheduler, manual_seed, wandb_init, \
    load_not_compatible_weights, demix_track, load_start_checkpoint, initialize_environment_ddp, initialize_environment, \
    get_lora
from dataset import MSSDataset, MSSValidationDataset
from torch.utils.data import DataLoader
from valid import valid_multi_gpu, prefer_target_instrument, valid
from metrics import sdr
from ml_collections import ConfigDict
from torch import nn
from tqdm.auto import tqdm
from losses import choice_loss
from accelerate import Accelerator
#from torchsummary import summary
from torchinfo import summary
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
# Tính mất mát của mô hình sau mỗi step (Hàm này hiện tại không dùng)
def forward_step(x, y, active_stem_ids, get_internal_loss, model, multi_loss, device_ids):
    if get_internal_loss:
        loss = model(x, y, active_stem_ids = active_stem_ids)
        if isinstance(device_ids, (list, tuple)):
            loss = loss.mean()
        return loss
    else:
        y_ = model(x)
        return multi_loss(y_, y, x)

def train_one_epoch(model: nn.Module, config: ConfigDict, args: argparse.Namespace, optimizer: torch.optim.Optimizer, device: torch.device, device_ids: List[int],
                    epoch: int, use_amp: bool, scheduler, gradient_accumulation_steps: int,train_loader: torch.utils.data.DataLoader, all_losses = None,
                    world_size = None, ema_model = None, accelerator: Accelerator = None) -> None:
    #ddp = True if world_size else False
    #should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    model.train().to(device)
    accelerator.print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]['lr']}')
    
    loss_val = 0.
    total = 0
    all_losses[f'epoch_{epoch}'] = []
    
    pbar = tqdm(train_loader, disable = not accelerator.is_main_process)
    normalize = getattr(config.training, 'normalize', False)

    get_internal_loss = not args.use_standard_loss

    for i, (batch, mixes) in enumerate(pbar):
        x = mixes #.to(device)
        y = batch #.to(device)

        if normalize:
            x, y = normalize_batch(x, y)
        loss = model(x, y)
        accelerator.backward(loss)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                    
            optimizer.step()

            if ema_model is not None:
                ema_model.update_parameters(accelerator.unwrap_model(model))

            optimizer.zero_grad() 
        
        li = loss.item() * gradient_accumulation_steps
        all_losses[f'epoch_{epoch}'].append(li)
        loss_val += li
        total += 1

        if accelerator.is_main_process:
            wandb.log({ 'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'total': total, 'loss_val': loss_val, 'i': i })
            pbar.set_postfix({ 'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1) })

    if accelerator.is_main_process:
        accelerator.print(f'Training loss: {(loss_val / total):.6f}')
        wandb.log({ 'train_loss': loss_val / total, 'epoch': epoch })

def compute_epoch_metrics(model: nn.Module, args: argparse.Namespace, config: ConfigDict, device: torch.device, device_ids: List[int], best_metric: float, 
                          epoch: int, scheduler: torch.optim.lr_scheduler, optimizer, all_time_all_metrics, all_losses, world_size = None, metrics_avg = None, 
                          all_metrics = None, scheduler_name = None) -> float:
    ddp = True if world_size else False
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    if not ddp:
        if torch.cuda.is_available() and len(device_ids) > 1:
            metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose = False)
        else:
            metrics_avg, all_metrics = valid(model, args, config, device, verbose = False)
        all_time_all_metrics[f'epoch_{epoch}'] = all_metrics

    metric_avg = metrics_avg[args.metric_for_scheduler]
    # Nếu giá trị đánh giá tốt hơn giá trị đánh giá trước đó, tạo file .ckpt và lưu lại thông số
    if metric_avg > best_metric:
        if args.each_metrics_in_name:
            stem_parts = []
            for stem_name, values in all_metrics[args.metric_for_scheduler].items():
                stem_values = np.array(values)
                mean_val = stem_values.mean()
                std_val = stem_values.std()
                stem_parts.append(f'{stem_name}_{args.metric_for_scheduler}_{mean_val:.4f}_std_{std_val:.4f}')
            stem_info = "__".join(stem_parts)
            store_path = (f'{args.result_path}/model_mel_band_roformer_ep_{epoch}_{stem_info}.ckpt')
        else:
            store_path = (f'{args.result_path}/model_mel_band_roformer_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt')
        if should_print:
            print(f'Store weights: {store_path}')
            save_weights(
                output_path = store_path,
                model = model,
                device_ids = device_ids,
                optimizer = optimizer,
                epoch = epoch,
                all_time_all_metrics = all_time_all_metrics,
                all_losses = all_losses,
                best_metric = best_metric,
                args = args, scheduler = scheduler 
            )
    # Lưu thông số sau mỗi epoch, đồng thời tạo file .ckpt chứa thông số tại thời điểm epoch đó
    if args.save_weights_every_epoch:
        metric_string = ''
        for m in metrics_avg:
            metric_string += f'_{m}_{metrics_avg[m]:.4f}'
        store_path = f'{args.result_path}/model_mel_band_roformer_ep_{epoch}{metric_string}.ckpt'
        save_weights(
            output_path = store_path,
            model = model,
            device_ids = device_ids,
            optimizer = optimizer,
            epoch = epoch,
            all_time_all_metrics = all_time_all_metrics,
            all_losses = all_losses,
            best_metric = best_metric,
            args = args,
            scheduler = scheduler
        )

    if scheduler_name in ['ReduceLROnPlateau']:
        scheduler.step(metric_avg)

    if should_print:
        wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
        for metric_name in metrics_avg:
            wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric

def save_weights(output_path: str, model: nn.Module, device_ids: List[int], optimizer: torch.optim.Optimizer, epoch: int, all_time_all_metrics, all_losses, 
                 best_metric: float, args,scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None) -> None:
    # Tạo một dict để lưu thông số và trạng thái của mô hình
    checkpoint: Dict[str, Any] = {
        'epoch': epoch,
        'optimizer_name': optimizer.__class__.__name__,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
        'all_metrics': all_time_all_metrics,
        'all_losses': all_losses
    }
    if args.train_lora_peft:
        model.save_pretrained(output_path + '_lora_')
    elif args.train_lora_loralib:
        import loralib as lora
        checkpoint['model_state_dict'] = lora.lora_state_dict(model)
    else:
        if torch.distributed.is_initialized():
            checkpoint['model_state_dict'] = model.module.state_dict()
        else:
            checkpoint['model_state_dict'] = (
                model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
            )
    # Lưu checkpoint vào trong file .ckpt của mô hình
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(checkpoint, output_path)
        print(f'Save checkpoint at {output_path}.')

def save_last_weights(args: argparse.Namespace, model: nn.Module, device_ids: List[int], optimizer: torch.optim.Optimizer, epoch: int, all_time_all_metrics, all_losses, 
                      best_metric: float, scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None) -> None:
    # File checkpoint (.ckpt) sẽ lưu lại
    store_path = f'{args.result_path}/last_mel_band_roformer.ckpt'
    save_weights(
        output_path = store_path, 
        model = model, 
        device_ids = device_ids, 
        optimizer = optimizer, 
        epoch = epoch, 
        all_time_all_metrics = all_time_all_metrics, 
        all_losses = all_losses, 
        best_metric = best_metric, 
        args = args,
        scheduler = scheduler
    )

def parse_args_train(dict_args: Union[argparse.Namespace, Dict, None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, help = 'Path to config file')
    parser.add_argument('--model_path', type = str, default = '', help = "Initial checkpoint to start training")
    parser.add_argument('--result_path', type = str, help = "Path to folder where results will be stored (weights, metadata)")
    parser.add_argument('--data_path', nargs = "+", type = str, help = "Dataset data paths. You can provide several folders.")
    parser.add_argument('--dataset_type', type = int, default = 1, help = "Dataset type. Must be one of: 1, 2, 3, 4, 5, 6, 7.")
    parser.add_argument('--valid_path', nargs = '+', type = str, help = 'Validation data paths. You can provide several folder.')
    parser.add_argument('--num_workers', type = int, default = 0, help = "Dataloader num_workers")
    parser.add_argument('--seed', type = int, default = 0, help = "Random seed")
    parser.add_argument('--device_ids', nargs = '+', type = int, default = [0], help = 'List of gpu ids')
    parser.add_argument('--validation_per_epochs', type = int, default = 1, help = 'Number of epochs for validation in period')
    parser.add_argument('--pin_memory', action = 'store_true', help = 'Dataloader pin memory')
    parser.add_argument('--persistent_workers', action = 'store_true', default = False, help = "dataloader persistent_workers")
    parser.add_argument('--prefetch_factor', type = int, default = None, help = 'Dataloader prefetch factor')
    parser.add_argument('--set_per_process_memory_fraction', action = 'store_true', help = 'Using only VRAM, no RAM')
    parser.add_argument('--safe_mode', action = 'store_true', help = 'Ignore forward errors')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')

    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--wandb_offline", action='store_true', help='Local wandb')

    parser.add_argument("--load_scheduler", action='store_true',help="Load scheduler state from checkpoint (if available)")
    parser.add_argument('--load_optimizer', action = 'store_true', help = 'Load optimizer state from checkpoint (if available)')
    parser.add_argument('--load_epoch', action = 'store_true', help = 'Load epoch number from checkpoint (if available)')
    parser.add_argument('--load_best_metric', action = 'store_true', help = 'Load best metric from checkpoint (if available)')
    parser.add_argument('--load_all_metrics', action = 'store_true', help = 'Load all metrics from checkpoints (if available)')
    parser.add_argument('--load_all_losses', action = 'store_true', help = 'Load all losses from checkpoint (if available)')
    parser.add_argument("--load_only_compatible_weights", action='store_true', help="using only VRAM, no RAM")

    parser.add_argument('--masked_loss_coef', type = float, default = 1, help = 'Coef for loss')
    parser.add_argument('--use_standard_loss', action = 'store_true', help = 'Roformers will use provided loss instead of internal')
    parser.add_argument("--loss", type = str, nargs = '+', 
        choices = ['masked_loss', 'mse_loss', 'l1_loss', 'multistft_loss', 'spec_masked_loss', 'spec_rmse_loss', 'log_wmse_loss', 'l1_snr_loss', 'l1_snr_db_loss', 'stft_l1_snr_db_loss', 'multi_l1_snr_db_loss'], 
        default = ['masked_loss'], help = 'List of loss functions to use')

    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"], 
        choices=['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness', 'l1_snr'], help='List of metrics to use.')
    parser.add_argument('--metric_for_scheduler', default = "sdr", 
        choices = ['k_sdr', 'sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness', 'l1_snr'], 
        help = 'Metric which will be used for scheduler.')
    parser.add_argument('--each_metrics_in_name', action = 'store_true', help = 'All stems in naming checkpoints')

    parser.add_argument("--train_lora_peft", action='store_true', help="Training with LoRA from peft")
    parser.add_argument("--train_lora_loralib", action='store_true', help="Training with LoRA from loralib")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument('--save_weights_every_epoch', action = 'store_true', help = 'Weights will be saved every epoch with all metric values')

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    get_internal_loss = not args.use_standard_loss

    if get_internal_loss:
        args.loss = ['mel_band_roformer_loss']

    return args

def train_model(args: Union[argparse.Namespace, None], rank = None, world_size = None) -> None:    
    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args_train(args)
    # Distributed Data Parallel (DDP) cho phép huấn luyện song song mô hình trên nhiều GPU hay thiết bị.
    ddp = True if world_size else False
    if ddp:
        initialize_environment_ddp(rank, world_size, args.seed, args.result_path)
    else:
        initialize_environment(args.seed, args.result_path)

    config = ConfigDict(yaml.load(open(args.config_path), Loader = yaml.FullLoader))
    model = get_model_from_config(config)
    accelerator.print(f"Instruments: {config.training.instruments}")
    # Lấy tham số use_amp cho phép hoặc không cho phép mixed precision.
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    
    if ddp:
        batch_size = config.training.batch_size
    else:   
        batch_size = config.training.batch_size * len(device_ids)
    # Thiết lập một tiến trình mới trong Weights & Biases (W&B) cho phép theo dõi quá trình thử nghiệm, ghi nhớ lại đánh giá và lưu lại các mẫu
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb_init(args, config, batch_size)
    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size = batch_size,
        metadata_path = os.path.join(args.result_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type = args.dataset_type,
        verbose = accelerator.is_main_process,
    )
    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = args.pin_memory)

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)

    valid_loader = DataLoader(validset, batch_size = 1, shuffle = False, )
    valid_loader = accelerator.prepare(valid_loader)

    if args.model_path:
        checkpoint = torch.load(args.model_path, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='train')

    model = get_lora(args, config, model)
    # Exponential Moving Average (EMA) là một loại trung bình động, giúp lọc nhiễu và tạo đường cong mượt mà
    ema_model = None
    # Nếu thông số của biến ema_momentum có trong config.yaml, cho phép mô hình giữ trung bình các tham số trong quá trình huấn luyện
    if hasattr(config.training, 'ema_momentum') and config.training.ema_momentum > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        accelerator.print(f"Initializing EMA with decay: {config.training.ema_momentum}")
        ema_model = AveragedModel(
            accelerator.unwrap_model(model), 
            multi_avg_fn = get_ema_multi_avg_fn(config.training.ema_momentum)
        )
        ema_model.to(device)
    # Đánh giá mô hình trước khi huấn luyện
    if args.pre_valid:
        model_to_valid = ema_model if ema_model is not None else model
        if ddp:
            valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose = False)
        else:
            if torch.cuda.is_available() and len(args.device_ids) > 1:
                valid_multi_gpu(model_to_valid, args, config, args.device_ids, verbose = True)
            else:
                valid(model_to_valid, args, config, device, verbose = True)
    
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    optimizer = get_optimizer(config, model, accelerator)

    scheduler = get_scheduler(config, optimizer)
    scheduler_name = scheduler.name

    if accelerator.is_main_process:
        if world_size:
            ef_batch_size = batch_size * gradient_accumulation_steps * world_size
        else:
            ef_batch_size = batch_size * gradient_accumulation_steps
        print(f'Processes GPU: {accelerator.num_processes}')
        print(f"Metrics for training: {args.metrics}\n"
            f"Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {config.training.patience}\n"
            f"Reduce factor: {config.training.reduce_factor}\n"
            f"Batch size: {batch_size}\n"
            f"Grad accum steps: {gradient_accumulation_steps}\n"
            f"Effective batch size: {ef_batch_size}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}")

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    accelerator.print(f'Train for: {config.training.num_epochs}')
    best_sdr = -100
    
    if args.model_path and 'optimizer_state_dict' in checkpoint and args.load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.model_path and "scheduler_state_dict" in checkpoint and args.load_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if args.model_path and "epoch" in checkpoint and args.load_epoch:
        start_epoch = checkpoint['epoch'] #+ 1
    else:
        start_epoch = 0

    if args.model_path and 'best_metric' in checkpoint and args.load_best_metric:
        best_metric = checkpoint['best_metric']
    else:
        best_metric = float('-inf')
    
    if args.model_path and 'all_metrics' in checkpoint and args.load_all_metrics:
        all_time_all_metrics = checkpoint['all_metrics']
    else:
        all_time_all_metrics = {}

    if args.model_path and 'all_losses' in checkpoint and args.load_all_losses:
        all_losses = checkpoint['all_losses']
    else:
        all_losses = {}

    losses_plot = list()
    if args.set_per_process_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    safe_mode = args.safe_mode
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    #summary(model, (config.audio.num_channels, config.audio.chunk_size), batch_size = batch_size)
    summary(model, (batch_size, config.audio.num_channels, config.audio.chunk_size), depth = 5)
    
    for epoch in range(start_epoch, config.training.num_epochs):
        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch + 1, use_amp, scheduler, gradient_accumulation_steps, train_loader, all_losses, 
                        world_size, ema_model, accelerator)

        model_to_valid = ema_model if ema_model is not None else model
        if should_print:
            save_last_weights(args, model, device_ids, optimizer, epoch + 1, all_time_all_metrics, all_losses, best_metric, scheduler)
        print()
        if epoch == start_epoch:
            for e in range(0, epoch + 1):
                loss_val = 0.
                total = 0
                for li in all_losses[f'epoch_{e + 1}']:
                    loss_val += li
                    total += 1
                losses_plot.append(loss_val / total)
        else:
            loss_val = 0.
            total = 0
            for li in all_losses[f'epoch_{epoch + 1}']:
                loss_val += li
                total += 1
            losses_plot.append(loss_val / total)
        epoch_range = list()
        for i in range(0, epoch + 1):
            epoch_range.append(i + 1)
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        ax.plot(epoch_range, losses_plot, label = "Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        fig.show()
        buf = io.BytesIO()
        fig.savefig(buf, format = 'png')

        buf.seek(0)
        img = Image.open(buf)
        image_path = args.result_path + f"loss_epoch_{epoch + 1}.png"
        img.save(image_path)
        print(f'Training loss plot was saved at {image_path}')
        
        accelerator.wait_for_everyone()

        NUMBER_OF_EPOCHS_IN_PERIOD = args.validation_per_epochs
        if (epoch + 1) % NUMBER_OF_EPOCHS_IN_PERIOD == 0: # % 100
            if ddp:
                metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose = False)
                if rank == 0:
                    all_time_all_metrics[f'epoch_{epoch + 1}'] = all_metrics
                    best_metric = compute_epoch_metrics(
                        model = model,
                        args = args,
                        config = config,
                        device = device,
                        device_ids = device_ids,
                        best_metric = best_metric,
                        epoch = epoch + 1,
                        scheduler = scheduler,
                        optimizer = optimizer,
                        all_time_all_metrics = all_time_all_metrics,
                        all_losses = all_losses,
                        world_size = world_size,
                        metrics_avg = metrics_avg,
                        all_metrics = all_metrics,
                        scheduler_name = scheduler_name
                    )    
            else:
                best_metric = compute_epoch_metrics(
                    model = model,
                    args = args,
                    config = config,
                    device = device,
                    device_ids = device_ids,
                    best_metric = best_metric,
                    epoch = epoch + 1,
                    scheduler = scheduler,
                    optimizer = optimizer,
                    all_time_all_metrics = all_time_all_metrics,
                    all_losses = all_losses,
                    scheduler_name = scheduler_name
                )
if __name__ == "__main__":
    train_model(None)
