r'''
cd D:\Code\projects\DDSP-SVC
$python="D:\Software\SVC-Fusion\project\.conda\python.exe"

$name='「少女」'
& $python train_reflow.py -n $name


& $python train_reflow.py -c configs/fritia.yaml
& $python train_reflow.py -c configs/acacia.yaml
& $python train_reflow.py -c configs/megumin.yaml
& $python train_reflow.py -c configs/kazuma.yaml
screen -S ddsp python train_reflow.py -c configs/reflow.yaml
screen -S ddsp python train_reflow.py -c configs/reflow_tune.yaml

tensorboard --logdir D:/Code/projects/DDSP-SVC/exp
'''

import argparse
import traceback
from pathlib import Path

import torch
# from torch.optim import lr_scheduler
import fairseq
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
from omegaconf import OmegaConf

from optimizer import lr_scheduler
# from optimizer.muon import Muon_AdamW
from optimizer.muon_moonshot import get_params_for_muon, Muon
from logger import utils
from reflow.data_loaders import get_data_loaders
from reflow.vocoder import Vocoder, Unit2Wav
from logger.saver import Saver
from logger.utils import DotDict


def handle_config(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help='name for this exp')
    parser.add_argument('-c', '--config', type=str, default=None, help='path to the config file (default: exp/$name/config.yaml)')
    # return parser.parse_args(args=args, namespace=namespace)
    args, unknown_args = parser.parse_known_args(args=args, namespace=namespace)
    if args.config is None:
        args.config = f'exp/{args.name}/config.yaml'
    
    # load config
    print(' > config:', args.config)
    cfg = OmegaConf.load(args.config)
    if cfg.env.resume_path and (resume_path := Path(cfg.env.resume_path)).is_file():
        resume_config = resume_path.with_suffix('.yaml')
        cfg = OmegaConf.merge(OmegaConf.load(resume_config), cfg)
    # unknown_args.extend()
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown_args))
    args = DotDict(**OmegaConf.to_container(cfg))
    return args

if __name__ == '__main__':
    # parse commands
    args = handle_config()
    # Saver 初始化时会自动保存args
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    if args.model.type == 'RectifiedFlow':
        from reflow.solver import train
        model = Unit2Wav(
            args.data.sampling_rate,
            args.data.block_size,
            args.model.win_length,
            args.data.encoder_out_channels, 
            args.model.n_spk,
            args.model.use_norm,
            args.model.use_attention,
            args.model.use_pitch_aug,
            vocoder.dimension,
            args.model.n_aux_layers,
            args.model.n_aux_chans,
            args.model.n_layers,
            args.model.n_chans
        )
                    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    # load parameters
    pm, po = get_params_for_muon(model)
    optimizer = Muon(args.train.lr, args.train.weight_decay, pm, adamw_params=po)
    # optimizer = Muon_AdamW(model, muon_args={'weight_decay': args.train.weight_decay}, adamw_args={'weight_decay': 0})
    global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    if global_step == 0 and args.env.resume_path is not None:
        # 尝试加载底模
        global_step, model, optimizer = utils.load_model(args.env.resume_path, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((global_step-2) // args.train.decay_step, 0)

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False, selected_audio_pattern=args.data.selected_audio_pattern)
    if args.train.max_steps is None:
        args['train']['max_steps'] = args.train.epochs * len(loader_train) + 1
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=global_step-2)
    scheduler = lr_scheduler.warmup_stage_decay(optimizer, args.train.decay_step, args.train.max_steps, decay_rate=args.train.gamma, last_steps=global_step)
    exp_name = args.env.expdir.split('/')[-1]
    # saver
    saver = Saver(args, initial_global_step=global_step)

    def melt_save():
        op = optimizer if args.train.save_opt else None
        saver.save_model(model, op, postfix=f'melt')
        print('已保存当前的模型权重及优化器状态...')
    try:
        print(f'{exp_name} 开始训练！')
        train(args, saver, model, optimizer, scheduler, vocoder, loader_train, loader_valid)
        print('结束训练！')
    except KeyboardInterrupt:
        print('\n检测到 Ctrl+C，正在退出程序...')
    except Exception:
        err_log = f'error_{exp_name}.log'
        traceback.print_exc(file=open(err_log, 'w', encoding='utf-8'))
        raise
    finally:
        melt_save()
