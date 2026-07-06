import importlib
import os
import torch
import numpy as np
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from setup import init_config, init_distributed


def main():
    # Handle Single-Process Execution
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"

    ddp_info = init_distributed(seed=777)
    device = ddp_info.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./assets", help="Where to save the canonical assets")
    parser.add_argument("--subject_idx", type=int, default=27, help="Index of the subject in dataset to process")
    args, unknown = parser.parse_known_args()

    config = init_config()
    if "inference" not in config: config.inference = {}
    config.inference.if_inference = True
    config.training.target_has_input = False 

    if ddp_info.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)

    # Load Dataset
    dataset_name = config.training.get("dataset_name", "data.dataset_human.HumanGSDataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)

    # Load Model (Dynamically instantiated based on config)
    module, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module).__dict__[class_name]
    model = ModelClass(config).to(device)
    model = DDP(model, device_ids=[ddp_info.local_rank])

    if ddp_info.is_main_process:
        print(f"Loading checkpoint: {config.training.checkpoint_dir}")

    model.module.load_ckpt(config.training.checkpoint_dir)
    model.eval()

    # Get Data
    if args.subject_idx >= len(dataset):
        if ddp_info.is_main_process:
            print(f"Error: subject_idx {args.subject_idx} is out of bounds.")
        return

    data = dataset[args.subject_idx]

    batch = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v 

    scene_name = batch.get('scene_name', f"subject_{args.subject_idx}")
    if isinstance(scene_name, torch.Tensor): 
        scene_name = str(scene_name.item())
    elif isinstance(scene_name, list):
        scene_name = str(scene_name[0])

    # Define AMP Type
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }
    amp_dtype = amp_dtype_mapping.get(config.training.amp_dtype, torch.float16)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=config.training.get("use_amp", False)):
        asset = model.module.predict_canonical_asset(batch)

    if ddp_info.is_main_process:
        save_path = os.path.join(args.save_dir, f"{scene_name}_asset.pt")
        save_dict = {k: v.squeeze(0).float().cpu() for k, v in asset.items()}
        torch.save(save_dict, save_path)

        print(f"Asset saved to {save_path}")
        print(f"Num Gaussians: {save_dict['means3D'].shape[0]}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()