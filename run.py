#!/usr/bin/env python3
"""Main entry point for SatNav training, evaluation, and inference.

This script provides a unified interface for running different modes:
- train: Train a model using imitation learning
- eval: Evaluate a trained model on a dataset split
- inference: Run inference to generate predictions

Usage:
    python run.py --exp-config CONFIG_PATH --run-type {train|eval|inference} [opts...]

Examples:
    # Training
    python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train
    
    # Evaluation
    python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval
    
    # Evaluation with config overrides
    python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \\
        EVAL.SPLIT val_unseen EVAL.CKPT_PATH data/checkpoints/seq2seq/ckpt.5.pth
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import satnav.training.dagger_trainer  # noqa: F401
import satnav.training.offline_trainer  # noqa: F401
import satnav.training.recollect_trainer  # noqa: F401
from satnav.training.registry import get_trainer
from satnav.training.distributed import cleanup_distributed, setup_distributed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SatNav VLN Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python run.py --exp-config configs/baselines/seq2seq.yaml --run-type train
  
  # Evaluation
  python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval
  
  # Override config from command line
  python run.py --exp-config configs/baselines/seq2seq.yaml --run-type eval \\
      EVAL.SPLIT val_unseen
        """
    )
    
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="Operation mode: train, eval, or inference"
    )
    
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (e.g., EVAL.SPLIT val_unseen)"
    )
    
    args = parser.parse_args()
    
    # Validate experiment config exists
    if not os.path.exists(args.exp_config):
        print(f"Error: Experiment config not found: {args.exp_config}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"SatNav VLN - {args.run_type.upper()} Mode")
    print("=" * 80)
    print(f"Experiment config: {args.exp_config}")
    
    # Load experiment configuration
    config = OmegaConf.load(args.exp_config)
    
    # Load default config if specified (for inheritance)
    # Support both _base_ (OmegaConf style) and BASE_CONFIG_PATH (custom)
    base_config_path = None
    if "_base_" in config:
        base_config_path = config._base_
    elif "BASE_CONFIG_PATH" in config:
        base_config_path = config.BASE_CONFIG_PATH
    
    if base_config_path:
        if os.path.exists(base_config_path):
            print(f"Loading base config: {base_config_path}")
            base_config = OmegaConf.load(base_config_path)
            # Merge: base config first, then experiment config (experiment config overrides)
            config = OmegaConf.merge(base_config, config)
            # Remove _base_ from final config to avoid confusion
            if "_base_" in config:
                del config._base_
            if "BASE_CONFIG_PATH" in config:
                del config.BASE_CONFIG_PATH
        else:
            print(f"Warning: Base config not found: {base_config_path}")
    
    # Load and merge task config if specified
    if "BASE_TASK_CONFIG_PATH" in config:
        task_config_path = config.BASE_TASK_CONFIG_PATH
        if os.path.exists(task_config_path):
            print(f"Loading task config: {task_config_path}")
            task_config = OmegaConf.load(task_config_path)
            # Merge: task config first, then experiment config (experiment config overrides)
            config = OmegaConf.merge(task_config, config)
        else:
            print(f"Warning: Task config not found: {task_config_path}")
    
    # Apply command line overrides
    if args.opts:
        print(f"Applying config overrides: {' '.join(args.opts)}")
        try:
            # Format opts as key=value pairs for OmegaConf
            # Convert ["KEY1", "VALUE1", "KEY2", "VALUE2"] to ["KEY1=VALUE1", "KEY2=VALUE2"]
            formatted_opts = []
            for i in range(0, len(args.opts), 2):
                if i + 1 < len(args.opts):
                    formatted_opts.append(f"{args.opts[i]}={args.opts[i+1]}")
                else:
                    print(f"Warning: Ignoring unpaired option: {args.opts[i]}")
            
            # Parse using formatted opts
            override_config = OmegaConf.from_dotlist(formatted_opts)
            
            # Merge overrides
            config = OmegaConf.merge(config, override_config)
            
        except Exception as e:
            print(f"Error parsing config overrides: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    try:
        setup_distributed(config, args.run_type)

        if (
            OmegaConf.select(config, "DISTRIBUTED.enabled", default=False)
            and args.run_type == "train"
            and not OmegaConf.select(config, "DISTRIBUTED.runtime_enabled", default=False)
        ):
            print("Distributed training requested, but WORLD_SIZE<=1. Falling back to single-process training.")

        if OmegaConf.select(config, "DISTRIBUTED.runtime_enabled", default=False):
            print(
                "Distributed runtime: "
                f"rank={config.DISTRIBUTED.rank}, "
                f"local_rank={config.DISTRIBUTED.local_rank}, "
                f"world_size={config.DISTRIBUTED.world_size}"
            )

        # Get trainer name from config
        if "TRAINER_NAME" not in config:
            print("Error: TRAINER_NAME not specified in config")
            sys.exit(1)

        trainer_name = config.TRAINER_NAME
        print(f"Trainer: {trainer_name}")
        print("=" * 80)

        # Get trainer class from registry
        try:
            trainer_class = get_trainer(trainer_name)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Instantiate trainer
        trainer = trainer_class(config)

        # Execute based on run_type
        if args.run_type == "train":
            trainer.train()
        elif args.run_type == "eval":
            trainer.eval()
        elif args.run_type == "inference":
            if hasattr(trainer, 'inference'):
                trainer.inference()
            else:
                print(f"Error: Trainer '{trainer_name}' does not support inference mode")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during {args.run_type}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_distributed()
    
    print("\n" + "=" * 80)
    print(f"{args.run_type.upper()} completed successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
