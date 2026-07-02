import argparse
import logging
import math
import os
import os.path as osp
import random
import sys
import copy
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

try:
    import wandb
except ImportError:
    wandb = None

# import open_clip

import options as option
from models import create_model

# Import open_clip from universal-restoration directory
_universal_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _universal_dir not in sys.path:
    sys.path.insert(0, _universal_dir)

import open_clip
from open_clip.prior_stage_model import PriorStageModel
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if (
            mp.get_start_method(allow_none=True) != "spawn"
    ):  
        mp.set_start_method("spawn", force=True) 
    rank = int(os.environ["RANK"])  
    num_gpus = torch.cuda.device_count()  
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # Auto-name runs by prior ablation switches and keep output dirs in sync.
    prior_switch_opt = opt.get("prior_switch", {})
    use_deg_prior = bool(prior_switch_opt.get("use_deg_prior", True))
    use_content_prior = bool(prior_switch_opt.get("use_content_prior", True))
    use_struct_prior = bool(prior_switch_opt.get("use_struct_prior", True))
    d = int(bool(prior_switch_opt.get("use_deg_prior", True)))
    c = int(bool(prior_switch_opt.get("use_content_prior", True)))
    s = int(bool(prior_switch_opt.get("use_struct_prior", True)))
    auto_name = f"ablation-d{d}-c{c}-s{s}"
    opt["name"] = auto_name

    # Keep UNet context switches aligned with prior switches to avoid
    # passing missing contexts into cross-attention blocks.n
    net_cfg = opt["network_G"]["setting"]
    net_cfg["use_degra_context"] = bool(net_cfg.get("use_degra_context", True) and use_deg_prior)
    net_cfg["use_image_context"] = bool(net_cfg.get("use_image_context", True) and use_content_prior)
    net_cfg["use_struct_context"] = bool(net_cfg.get("use_struct_context", True) and use_struct_prior)

    root = opt["path"]["root"]
    config_dir = osp.basename(osp.dirname(opt["path"]["experiments_root"]))
    experiments_root = osp.join(root, "experiments", config_dir, auto_name)
    opt["path"]["experiments_root"] = experiments_root
    opt["path"]["models"] = osp.join(experiments_root, "models")
    opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
    opt["path"]["log"] = experiments_root
    opt["path"]["val_images"] = osp.join(experiments_root, "val_images")

    opt = option.dict_to_nonedict(opt)
    wandb_run = None
    profile_timing = bool(opt["logger"].get("profile_timing", False))
    profile_interval = int(opt["logger"].get("profile_interval", opt["logger"]["print_freq"]))

    seed = opt["train"]["manual_seed"]

    if args.launcher == "none":  
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  
        rank = torch.distributed.get_rank()  

    torch.backends.cudnn.benchmark = True


    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  
    else:
        resume_state = None

    if rank <= 0:  
        if resume_state is None:
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                       and "pretrain_model" not in key
                       and "resume" not in key
                       and "tpgd" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
        use_wandb = opt.get("use_wandb", True)
        if use_wandb and wandb is not None:
            wandb_project = opt.get("wandb_project", "TPGDiff_universal_restoration")
            wandb_run = wandb.init(project=wandb_project, name=opt["name"])
        elif use_wandb and wandb is None:
            logger.warning("wandb is not installed, skip wandb logging.")
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    dataset_ratio = 200  
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    model = create_model(opt)
    device = model.device

    need_clip_prior = use_deg_prior or use_content_prior
    prior_model = None

    if rank <= 0:
        logger.info(
            "Prior switches | deg: %s, content: %s, struct: %s",
            use_deg_prior, use_content_prior, use_struct_prior
        )

    if need_clip_prior:
        prior_ckpt = opt["path"].get("prior", None)
        if prior_ckpt is None:
            raise ValueError("opt['path']['prior'] must be set when deg/content prior is enabled.")

        base_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
            device=device, precision="fp32"
        )
        teacher_encoder = base_model.visual
        student_encoder = copy.deepcopy(base_model.visual)
        deg_backbone = copy.deepcopy(base_model.visual)

        if hasattr(base_model.visual, "output_dim"):
            embed_dim = base_model.visual.output_dim
        elif hasattr(base_model, "embed_dim"):
            embed_dim = base_model.embed_dim
        else:
            raise RuntimeError("Cannot infer embed_dim from base_model.visual, please check model definition.")

        num_degradations = len(opt["distortion"])

        prior_model = PriorStageModel(
            teacher_encoder=teacher_encoder,
            student_encoder=student_encoder,
            deg_backbone=deg_backbone,
            embed_dim=embed_dim,
            num_degradations=num_degradations,
            content_loss_weight=1.0,
            deg_loss_weight=1.0,
            use_cosine_distill=True,
            normalize_embedding=True,
            freeze_teacher=True,
            freeze_deg_backbone=True,
        ).to(device)

        ckpt = torch.load(prior_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        if len(state_dict) and next(iter(state_dict.keys())).startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        prior_model.load_state_dict(state_dict, strict=True)
        prior_model.eval()

    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state) 
    else:
        current_step = 0
        start_epoch = 0

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                     eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    if profile_timing and rank <= 0:
        logger.info("Timing profiler enabled. Note: CUDA sync is used and introduces overhead.")

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    prev_iter_end = time.perf_counter()
    timing_acc = {"data": 0.0, "prior": 0.0, "sde": 0.0, "optimize": 0.0, "lr": 0.0, "iter": 0.0}
    timing_cnt = 0

    os.makedirs('image', exist_ok=True)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            iter_start = time.perf_counter()
            data_time = max(0.0, iter_start - prev_iter_end)
            current_step += 1

            if current_step > total_iters:
                break

            LQ, GT, deg_type = train_data["LQ"], train_data["GT"], train_data["type"]
            img4clip = train_data["LQ_clip"].to(device) if need_clip_prior else None
            prior_t0 = time.perf_counter()
            if need_clip_prior:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    content_context = (
                        prior_model.get_content_prior(img4clip).float()
                        if use_content_prior else None
                    )  # [B, D]
                    deg_context = (
                        prior_model.encode_for_degradation(img4clip).float()
                        if use_deg_prior else None
                    )  # [B, D]
            else:
                content_context = None
                deg_context = None
            if profile_timing:
                torch.cuda.synchronize(device)
            prior_time = time.perf_counter() - prior_t0

            sde_t0 = time.perf_counter()
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)
            if profile_timing:
                torch.cuda.synchronize(device)
            sde_time = time.perf_counter() - sde_t0

            model.feed_data(states, LQ, GT, deg_context=deg_context, content_context=content_context)  # xt, mu, x0
            opt_t0 = time.perf_counter()
            model.optimize_parameters(current_step, timesteps, sde)
            if profile_timing:
                torch.cuda.synchronize(device)
            optimize_time = time.perf_counter() - opt_t0
            lr_t0 = time.perf_counter()
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            lr_time = time.perf_counter() - lr_t0
            iter_time = time.perf_counter() - iter_start
            prev_iter_end = time.perf_counter()

            if profile_timing and rank <= 0:
                timing_acc["data"] += data_time
                timing_acc["prior"] += prior_time
                timing_acc["sde"] += sde_time
                timing_acc["optimize"] += optimize_time
                timing_acc["lr"] += lr_time
                timing_acc["iter"] += iter_time
                timing_cnt += 1
                if current_step % profile_interval == 0 and timing_cnt > 0:
                    logger.info(
                        "[timing] iter_avg={:.4f}s data={:.4f}s prior={:.4f}s sde={:.4f}s optimize={:.4f}s lr={:.4f}s ({} iters)".format(
                            timing_acc["iter"] / timing_cnt,
                            timing_acc["data"] / timing_cnt,
                            timing_acc["prior"] / timing_cnt,
                            timing_acc["sde"] / timing_cnt,
                            timing_acc["optimize"] / timing_cnt,
                            timing_acc["lr"] / timing_cnt,
                            timing_cnt,
                        )
                    )
                    timing_acc = {"data": 0.0, "prior": 0.0, "sde": 0.0, "optimize": 0.0, "lr": 0.0, "iter": 0.0}
                    timing_cnt = 0

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                train_wandb_log = {
                    "train/epoch": epoch,
                    "train/iter": current_step,
                    "train/lr": model.get_current_learning_rate(),
                }
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    train_wandb_log[f"train/{k}"] = float(v)
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
                    if wandb_run is not None:
                        wandb.log(train_wandb_log, step=current_step)

            if current_step % opt["train"]["val_freq"] == 0:
                if rank <= 0:
                    torch.cuda.empty_cache()
                avg_psnr = 0.0
                avg_val_loss = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    LQ, GT, deg_type = val_data["LQ"], val_data["GT"], val_data["type"]
                    img4clip = val_data["LQ_clip"].to(device) if need_clip_prior else None
                    if need_clip_prior:
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            content_context = (
                                prior_model.get_content_prior(img4clip).float()
                                if use_content_prior else None
                            )
                            deg_context = (
                                prior_model.encode_for_degradation(img4clip).float()
                                if use_deg_prior else None
                            )
                    else:
                        content_context = None
                        deg_context = None

                    timesteps_val, states_val = sde.generate_random_states(x0=GT, mu=LQ)
                    model.feed_data(states_val, LQ, GT, deg_context=deg_context, content_context=content_context)
                    with torch.no_grad():
                        sde.set_mu(model.condition)
                        timesteps_val = timesteps_val.to(device)
                        struct_tokens = model.struct_prior(model.condition) if model.struct_prior is not None else None
                        noise = sde.noise_fn(
                            model.state,
                            timesteps_val.reshape(-1),
                            deg_context=model.deg_context,
                            content_context=model.content_context,
                            struct_tokens=struct_tokens,
                        )
                        score = sde.get_score_from_noise(noise, timesteps_val)
                        xt_1_expection = sde.reverse_sde_step_mean(model.state, score, timesteps_val)
                        xt_1_optimum = sde.reverse_optimum_step(model.state, model.state_0, timesteps_val)
                        val_loss = model.weight * model.loss_fn(xt_1_expection, xt_1_optimum)
                        avg_val_loss += float(val_loss.item())

                    noisy_state = sde.noise_state(LQ)

                    model.feed_data(noisy_state, LQ, GT, deg_context=deg_context, content_context=content_context)
                    model.test(sde, struct_tokens=struct_tokens)

                    if rank <= 0:
                        visuals = model.get_current_visuals()

                        output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                        gt_img = util.tensor2img(GT.squeeze())  # uint8
                        lq_img = util.tensor2img(LQ.squeeze())

                        util.save_img(output, f'image/{idx}_{deg_type[0]}_SR.png')
                        util.save_img(gt_img, f'image/{idx}_{deg_type[0]}_GT.png')
                        util.save_img(lq_img, f'image/{idx}_{deg_type[0]}_LQ.png')

                        # calculate PSNR
                        avg_psnr += util.calculate_psnr(output, gt_img)
                    idx += 1

                    if idx > 99:
                        break

                if rank <= 0:
                    avg_psnr = avg_psnr / idx
                    avg_val_loss = avg_val_loss / idx

                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        best_iter = current_step

                    logger.info("# Validation # PSNR: {:.6f}, val_loss: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(
                        avg_psnr, avg_val_loss, best_psnr, best_iter
                    ))
                    logger_val = logging.getLogger("val") 
                    logger_val.info("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, val_loss: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_val_loss
                    ))
                    print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, val_loss: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_val_loss
                    ))
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)
                        tb_logger.add_scalar("val_loss", avg_val_loss, current_step)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/psnr": float(avg_psnr),
                                "val/loss": float(avg_val_loss),
                                "val/val_loss": float(avg_val_loss),
                                "val/best_psnr": float(best_psnr),
                                "epoch": int(epoch),
                                "iter": int(current_step),
                            },
                            step=current_step,
                        )

            if error.value:
                sys.exit(0)
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
        if wandb_run is not None:
            wandb_run.finish()
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            tb_logger.close()


if __name__ == "__main__":
    main()
