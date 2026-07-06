import json
import os
import random
import os
import argparse


def generate_indices(dataset_path, output_path, num_input, num_target, seed=777):
    random.seed(seed)
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    view_indices = {}
    for subject in data:
        scene_name = subject['scene_name']
        frames = subject['frames']
        total_frames = len(frames)
        if total_frames < (num_input + num_target):
            print(f"Skipping {scene_name}: Not enough frames ({total_frames})")
            view_indices[scene_name] = None
            continue

        all_indices = list(range(total_frames))
        context_indices = sorted(random.sample(all_indices, num_input))
        remaining = [i for i in all_indices if i not in context_indices]
        if len(remaining) < num_target:
            target_indices = sorted(random.sample(all_indices, num_target))
        else:
            target_indices = sorted(random.sample(remaining, num_target))
        view_indices[scene_name] = { "context": context_indices, "target": target_indices }

    print(f"Generated indices for {len(view_indices)} scenes.")
    with open(output_path, 'w') as f:
        json.dump(view_indices, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to your test.json")
    parser.add_argument("--output", type=str, default="fixed_eval_views_thuman21_1.json")
    parser.add_argument("--input_views", type=int, default=1)
    parser.add_argument("--target_views", type=int, default=4)
    args = parser.parse_args()
    generate_indices(args.test_json, args.output, args.input_views, args.target_views)
