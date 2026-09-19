from model import MelBandRoformer
from typing import Union, Dict, Any, List, Tuple, Callable, Optional
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, AdamW, RAdam, RMSprop
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
import time
import numpy as np
import sys
import argparse
import os
import soundfile as sf
import json
import random
import wandb
import librosa.display
import matplotlib.pyplot as plt

def get_model_from_config(config):
    return MelBandRoformer(**dict(config.model))
# Ghi lại quá trình đánh giá vào trong log
def logging(logs: List[str], text: str, verbose_logging: bool = False) -> Union[List[str], None]:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(text)
        if verbose_logging:
            logs.append(text)
    return logs
# Ghi lại kết quả các chỉ số đánh giá vào file results.txt
def write_results_in_file(output_dir: str, logs: List[str]) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open(f'{output_dir}/results.txt', 'w') as out:
            for item in logs:
                out.write(item + '\n')
# Tạo mảng phân chia cửa sổ
def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein

    return window.to(device)
# Thực hiện việc phân tách file âm thanh
def demix_track(config, model, mix, device, first_chunk_time = None):
    C = config.inference.chunk_size
    N = config.inference.num_overlap
    step = C // N
    fade_size = C // 10
    border = C - step

    mix = torch.tensor(mix, dtype = torch.float32)

    if mix.shape[1] > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    windowing_array = get_windowing_array(C, fade_size, device)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else: 
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)
            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)

            i = 0
            total_length = mix.shape[1]
            num_chunks = (total_length + step - 1) // step

            if first_chunk_time is None:
                start_time = time.time()
                first_chunk = True
            else:
                start_time = None
                first_chunk = False

            while i < total_length:
                part = mix[:, i: i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input = part, pad = (0, C - length), mode = 'reflect')
                    else:
                        part = nn.functional.pad(input = part, pad = (0, C - length, 0, 0), mode = 'constant', value = 0)
                if first_chunk and i == 0:
                    chunk_start_time = time.time()

                x = model(part.unsqueeze(0))[0]
                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window[-fade_size:] = 1

                result[..., i:i+length] += x[..., :length] * window[..., :length]
                counter[..., i:i+length] += window[..., :length]
                i += step

                if first_chunk and i == step:
                    chunk_time = time.time() - chunk_start_time
                    first_chunk_time = chunk_time
                    estimated_total_time = chunk_time * num_chunks
                    print(f"\nEstimated total processing time for this track: {estimated_total_time:.2f} seconds")
                    first_chunk = False

                if first_chunk_time is not None and i > step:
                    chunks_processed = i // step
                    time_remaining = first_chunk_time * (num_chunks - chunks_processed)
                    sys.stdout.write(f"\rEstimated time remaining: {time_remaining:.2f} seconds")
                    sys.stdout.flush()
            print()
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)
            
            if mix.shape[1] > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]
    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}, first_chunk_time
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}, first_chunk_time
# Khởi tạo mô hình và thiết bị hoặc GPU (Hàm này hiện tại không dùng)
def initialize_model_and_device(model: nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], nn.Module]:
    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda: {device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda: {device_ids[0]}')
            model = nn.DataParallel(model, device_ids = device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print('CUDA is not available. Running on CPU.')

    return device, model

def get_scheduler(config, optimizer):
    # Đặt scheduler mặc định là ReduceLROnPlateau, giảm learning rate mỗi khi sự tiến bộ của chỉ số đánh giá bị trì hoãn (Ví dụ, liên tục dao động tại một khu vực giá trị mà không giảm)
    scheduler_name = config.training.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_name == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience = config.training.get('patience', 10), factor = config.training.get('reduce_factor', 0.5))
    scheduler.name = scheduler_name
    return scheduler

def get_optimizer(config: ConfigDict, model: nn.Module, accelerator) -> torch.optim.Optimizer:
    #should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    optim_params = dict()
    
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        accelerator.print(f'Optimizer params from config:\n{optim_params}')

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr = config.training.lr, **optim_params)
    elif config.training.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr = config.training.lr, **optim_params)
    elif config.training.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr = config.training.lr, **optim_params)
    else:
        accelerator.print(f'Unknown optimizer: {config.training.optimizer}')
        exit()
    return optimizer
# Chuẩn hóa Tensor phần âm thanh thành phần (vocals, other)
def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y
# Tải các thông số không tương thích với mô hình trong điều kiện
def load_not_compatible_weights(model: nn.Module, old_model: dict, verbose: bool = False) -> None:
    should_print = verbose and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
    new_model = model.state_dict()

    if 'state' in old_model: 
        old_model = old_model['state']
    if 'state_dict' in old_model:
        old_model = old_model['state_dict']
    if 'model_state_dict' in old_model:
        old_model = old_model['model_state_dict']

    for el in new_model:
        if el in old_model:
            if should_print:
                print(f'Match found for {el}')
            if new_model[el].shape == old_model[el].shape:
                if should_print:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape) and should_print:
                    print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if should_print:
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype = np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if should_print:
                print(f'Match not found for {el}!')
    model.load_state_dict(new_model)
# Lấy các thông số LoRA từ config.yaml (Hàm này hiện tại không dùng)
def get_lora(args, config, model):
    if args.train_lora_loralib:
        import loralib as lora
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)
    if args.train_lora_peft:
        if args.lora_checkpoint_peft:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora_checkpoint_peft)
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
        else:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(**config['lora'])
            model = get_peft_model(model, lora_config)
    return model
# Tải các thông số LoRA dùng để fine-tune mô hình (Hàm này hiện tại không dùng)
def load_lora_weights(model: nn.Module, lora_path: str, device: str = 'cpu') -> None:
    lora_state_dict = torch.load(lora_path, map_location = device)
    model.load_state_dict(lora_state_dict, strict = False)
# Tải file checkpoint (.ckpt)
def load_start_checkpoint(args: argparse.Namespace, model: nn.Module, old_model, type_: str = 'train') -> None:
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    if should_print:
        print(f'Start from checkpoint: {args.model_path}')
    if type_ in ['train']:
        if not args.load_only_compatible_weights:
            load_not_compatible_weights(model, old_model, verbose = False)
        else:
            model.load_state_dict(torch.load(args.model_path))
    else:
        #device = 'cpu'
        if 'state' in old_model:
            old_model = old_model['state']
        if 'state_dict' in old_model:
            old_model = old_model['state_dict']
        if 'model_state_dict' in old_model:
            old_model = old_model['model_state_dict']
        model.load_state_dict(old_model)

    if args.lora_checkpoint_loralib:
        if should_print:
            print(f'Loading LoRa weights from: {args.lora_checkpoint_loralib}')
        load_lora_weights(model, args.lora_checkpoint_loralib)

# Thêm các thông số LoRA vào mô hình (Hàm này hiện tại không dùng)
def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    import loralib as lora
    replaced_layers = 0
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]

        if isinstance(module, nn.Linear):
            try:
                parent_module = model
                for submodule_name in hierarchy[:-1]:
                    parent_module = getattr(parent_module, submodule_name)
                setattr(
                    parent_module, 
                    layer_name, 
                    lora.Linear(in_features = module.in_features, out_features = module.out_features, bias = module.bias is not None, **config['lora'])
                )
                replaced_layers += 1
            except Exception as e:
                if should_print:
                    print(f'Error replacing layer {name}: {e}')
    
    if replaced_layers == 0 and should_print:
        print('Warning: No layers were replaced. Check the model structure and configuration.')
    elif should_print:
        print(f'Number of layers replaced with LoRA: {replaced_layers}')

    return model
# Tải thông tin tóm tắt về cấu trúc mô hình (Hàm này hiện tại không dùng)
def log_model_info(model: nn.Module, result_path = None):
    model_info = {
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__,
        'model_module': model.__class__.__module__,
    }
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info['parameters'] = {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_millions': round(total_params / 1e6, 2),
        'trainable_millions': round(trainable_params / 1e6, 2),
    }
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    model_info['memory'] = {
        'parameters_mb': round(param_size / 1024 / 1024, 2),
        'buffers_mb': round(buffer_size / 1024 / 1024, 2),
        'total_mb': round(model_size_mb, 2),
    }

    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                layer_info.append({ 'name': name, 'type': module.__class__.__name__, 'parameters': layer_params, })
    model_info['layers'] = layer_info

    if result_path:
        path = os.path.join(result_path, 'model_info.json')
        with open(path, 'w') as f:
            json.dump(model_info, f, indent = 2)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f'Model: {model_info['model_class']}')
        print(f'Total parameters: {model_info['parameters']['total']:,} ({model_info['parameters']['total_millions']}M)')
        print(f'Trainable parameters: {model_info['parameters']['trainable']:,} ({model_info['parameters']['trainable_millions']}M)')
        print(f'Model size: {model_info['memory']['total_mb']:.2f} MB')
        print(f'Number of layers: {len(layer_info)}')
# Đặt seed để tạo ra mã số ngẫu nhiên cho tất cả thiết bị hoặc GPU
def manual_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    os.environ['PYTHONHASHSEED'] = str(seed)
# Khởi tạo một môi trường DDP cho phép chạy quá trình trên nhiều GPU hoặc thiết bị
def initialize_environment_ddp(rank: int, world_size: int, seed: int = 0, result_path: str = None) -> None:
    seed = (seed + int(time.time())) % 55535 + 10000
    setup_ddp(rank, world_size, seed)
    manual_seed(seed)
    try:
        torch.multiprocessing.set_start_method('spawn', force = True)
    except RuntimeError as e:
        if 'contect has already been set' not in str(e):
            raise e
    if not (result_path is None):
        os.makedirs(result_path, exist_ok = True)
# Khởi tạo 1 môi trường chỉ chạy trên 1 GPU hoặc thiết bị
def initialize_environment(seed: int, result_path: str) -> None:
    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(result_path, exist_ok = True)
# Cài đặt môi trường DDP
def setup_ddp(rank: int, world_size: int, seed: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(seed)
    os.environ['USE_LIBUV'] = '0'
    try:
        torch.distributed.init_process_group(
            'nccl', rank = rank, world_size = world_size)
    except:
        torch.distributed.init_process_group(
            'gloo', rank = rank, world_size = world_size)
        if torch.distributed.get_rank() == 0:
            print(f"NCCL are not available. Using 'gloo' backend.")
    torch.cuda.set_device(rank)

def cleanup_ddp() -> None:
    torch.distributed.destroy_process_group()

def gen_wandb_name(args, config) -> str:
    instrum = '-'.join(config['training']['instruments'])
    time_str = time.strftime('%Y-%m-%d')
    name = f'mel_band_roformer_[{instrum}]_{time_str}'
    return name
# Cấu hình WanDB (Weights & Biases) hỗ trợ theo dõi quá trình thử nghiệm tác vụ Deep Learning
def wandb_init(args: argparse.Namespace, config: Union[ConfigDict, OmegaConf], batch_size: int) -> None:
    if args.wandb_offline:
        wandb.init(
            model = 'offline', 
            project = 'msst', 
            name = gen_wandb_name(args, config), 
            config = { 'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size }
        )
    elif args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode = 'disabled')
    else:
        wandb.login(key = args.wandb_key)
        wandb.init(
            project = 'msst', 
            name = gen_wandb_name(args, config), 
            config = { 'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size }
        )
# Đọc dữ liệu hoán vị của audio từ từng thư mục bài hát trong tập dữ liệu
def read_audio_transposed(path: str, instr: Optional[str] = None, skip_err: bool = False) -> Tuple[Optional[np.ndarray], Optional[int]]:
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            if should_print:
                print(f'No stem {instr}: skip!')
            return None, None
        else:
            raise RuntimeError(f'Error reading the file at {path}: {e}')
    else:
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis = -1)
        return mix.T, sr

def bigshifts_wrapper(config: ConfigDict, model: nn.Module, mix: torch.Tensor, device: torch.device, pbar: bool = False, 
                      bigshifts: int = 1) -> Union[Dict[str, np.ndarray], np.ndarray]:
    should_print = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    if bigshifts <= 0:
        bigshifts = 1

    if isinstance(mix, torch.Tensor):
        mix = mix.detach().cpu().numpy()

    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []

    if pbar and should_print:
        shifts_iterator = tqdm(shifts, decs = 'BigShifts passes...', leave = False)
    else: 
        shifts_iterator = shifts

    for shift in shifts_iterator:
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis = -1)
        sources = demix_track(config, model, shifted_mix, device, pbar)
        if isinstance(sources, dict):
            unshifted = { k: np.concatenate((v[..., shift:], v[..., :shift]), axis = -1) for k, v in sources.items() }
            results.append(unshifted)
        elif isinstance(sources, np.ndarray):
            unshifted = np.concatenate((sources[..., shift:], sources[..., :shift]), axis = -1)
            results.append(unshifted)
        else:
            raise ValueError('Unsupported return type from demix')

    if isinstance(results[0], dict):
        avg_result = {}
        for k in results[0]:
            avg_result[k] = np.mean([r[k] for r in results], axis = 0)
        return avg_result
    return np.mean(results, axis = 0)
# Áp dụng kỹ thuật TTA (Test-Time Augmentation) để chuyển đổi dữ liệu waveform gốc này ở khoảng thời gian suy luận nhằm cải thiện hiệu suất mô hình
def apply_tta(config, model: nn.Module, mix: torch.Tensor, waveforms_orig: Union[dict[str, np.ndarray], np.ndarray], device: torch.device, bigshifts: int = 1, 
              pbar: bool = False) -> Union[dict[str, np.ndarray], np.ndarray]:
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = bigshifts_wrapper(config, model, augmented_mix, device, bigshifts = bigshifts, pbar = pbar)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1
    return waveforms_orig
# Đảo ngược quá trình chuẩn hóa audio
def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    return audio * norm_params['std'] + norm_params['mean']
# Vẽ spectrogram (quang phổ) của waveform âm thanh đã được phân tách và waveform của cả bài hát (Hàm này hiện tại không dùng)
def draw_2_mel_spectrogram(estimates_waveform: np.ndarray, track_waveform: np.ndarray, sample_rate: int, length: float, output_base: str) -> None:
    waveforms = [estimates_waveform, track_waveform]
    titles = ['Estimates', 'Original']
    processed_waveforms: list[tuple[np.ndarray, int]] = []

    for waveform in waveforms:
        mono_signal = waveform.mean(axis = -1) if len(waveform.shape) > 1 else waveform
        if len(mono_signal) > 60 * sample_rate:
            mono_signal = mono_signal[::2]
            effective_sr = sample_rate // 2
        else:
            effective_sr = sample_rate
        processed_waveforms.append((mono_signal, effective_sr))

    fig_spec, axes_spec = plt.subplots(2, 1, figsize = (16, 10))

    for i, ((mono_signal, effective_sr), title) in enumerate(zip(processed_waveforms, titles)):
        S = librosa.feature.melspectrogram(y = mono_signal, sr = effective_sr, n_mels = 128)
        S_db = librosa.power_to_db(S, ref = np.max)
        img = librosa.display.specshow(S_db, cmap = 'plasma', sr = effective_sr, x_axis = 'time', y_axis = 'mel', ax = axes_spec[i])
        axes_spec[i].set_title(f'Mel-spectrogram: {title}', fontsize = 14, fontweight = 'bold')
        axes_spec[i].set_xlabel('Time (seconds)', fontsize = 12)
        axes_spec[i].set_ylabel('Frequency (mel)', fontsize = 12)
    fig_spec.suptitle(f'Mel-spectrograms: {os.path.basename(output_base)}', fontsize = 16, fontweight = 'bold', y = 0.98)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.94, hspace = 0.4, right = 0.88)
    spec_output = f'{output_base}_spectrograms.jpg'
    plt.savefig(spec_output, dpi = 150, bbox_inches = 'tight')
    plt.close(fig_spec)
    fig_wave, axes_wave = plt.subplots(2, 1, figsize = (16, 8))
    for i, ((mono_signal, effective_sr), title) in enumerate(zip(processed_waveforms, titles)):
        time = np.linspace(0, len(mono_signal) / effective_sr, len(mono_signal))
        if len(mono_signal) > 100000:
            plot_indices = np.arange(0, len(mono_signal), 10)
            axes_wave[i].plot(time[plot_indices], mono_signal[plot_indices], color = '#00ff88', alpha = 0.9, linewidth = 0.5)
        else:
            axes_wave[i].plot(time, mono_signal, color = '#00ff88', alpha = 0.9, linewidth = 0.8)
        axes_wave[i].fill_between(time, mono_signal, alpha = 0.3, color = '#00ff8833')
        axes_wave[i].set_xlabel('Time (seconds)', fontsize = 12)
        axes_wave[i].set_ylabel('Amplitude', fontsize = 12)
        axes_wave[i].set_title(f'Waveform: {title}', fontsize = 14, fontweight = 'bold')
        axes_wave[i].grid(True, alpha = 0.3, color = 'gray')
        axes_wave[i].set_xlim(0, time[-1])
    fig_wave.suptitle(f'Waveforms: {os.path.basename(output_base)}', fontsize = 16, fontweight = 'bold', y = 0.98)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.94, hspace = 0.4)
    wave_output = f'{output_base}_waveforms.jpg'
    plt.savefig(wave_output, dpi = 150, bbox_inches = 'tight')
    plt.close(fig_wave) 