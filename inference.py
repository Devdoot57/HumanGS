# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Adapted to implement HumanRAM (Yu et al., 2025).
# This code is an unofficial implementation of the paper "HumanRAM: Feed-forward Human Reconstruction and Animation Model using Transformers".

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation


def main():
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    # Set up DDP training/inference and Fix random seed
    # Fixed seed ensures 'random' view selection is deterministic for testing
    ddp_info = init_distributed(seed=777) 
    dist.barrier()

    # Set up tf32
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }

    # Define Output Directory
    if "inference_out_dir" not in config:
        config.inference_out_dir = os.path.join(
            os.path.dirname(config.training.checkpoint_dir), "inference_results"
        )
    
    if ddp_info.is_main_process:
        os.makedirs(config.inference_out_dir, exist_ok=True)
        print(f"Running inference; save results to: {config.inference_out_dir}")

    # Load data
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    
    if "inference" not in config:
        config.inference = {}
    config.inference.if_inference = True
    
    dataset = Dataset(config)

    datasampler = DistributedSampler(dataset, shuffle=False) # No shuffle for inference
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
        drop_last=False, # Don't drop last for inference
        sampler=datasampler
    )

    dist.barrier()

    # Import model and load checkpoint
    module, class_name = config.model.class_name.rsplit(".", 1)
    LVSM = importlib.import_module(module).__dict__[class_name]
    model = LVSM(config).to(ddp_info.device)

    if ddp_info.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n" + "="*40)
        print("  HumanGS Parameter Count")
        print("="*40)
        print(f"Total Parameters:     {total_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        print("="*40 + "\n")
    
    model = DDP(model, device_ids=[ddp_info.local_rank])
    
    model.module.load_ckpt(config.training.checkpoint_dir)

    if ddp_info.is_main_process:  
        import lpips
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

    dist.barrier()

    datasampler.set_epoch(0)
    model.eval()

    print(f"[Rank {ddp_info.local_rank}] Starting inference on {len(dataset)} samples.")

    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
            
            # Run Reconstruction (Forward Pass)
            result = model(batch)
            
            # Run Animation/Orbit (Optional Video Render)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            
            # Save Results
            export_results(result, config.inference_out_dir, inference_config=config.inference)
            
            if i % 10 == 0 and ddp_info.is_main_process:
                print(f"Processed {i}/{len(dataloader)} batches.")

            torch.cuda.empty_cache()

    dist.barrier()

    # Summarize metrics
    if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
        summarize_evaluation(config.inference_out_dir)
        
    dist.barrier()
    dist.destroy_process_group()
    print("Inference finished.")


if __name__ == "__main__":
    main()