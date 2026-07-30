import argparse
import sys
from src.config import TrainConfig
from src.yaml_config_loader import load_train_config_from_yaml


def main():
    parser = argparse.ArgumentParser(
        description="fine-tuning-lab: CLI entrypoint for experiment management"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Run training with specified config YAML"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML"
    )
    train_parser.add_argument(
        "--dry",
        action="store_true",
        help="Print config and exit (no training)"
    )

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Print dataset stats from config YAML"
    )
    stats_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML"
    )

    args = parser.parse_args()

    if args.command == "train":
        cfg = load_train_config_from_yaml(args.config)
        print("Loaded config:")
        try:
            import yaml
            print(yaml.dump(cfg.__dict__, sort_keys=False, default_flow_style=False))
        except ImportError:
            print(cfg)
        if args.dry:
            sys.exit(0)
        print("Training run stub (actual training not implemented in CLI yet).")
        # Placeholder: integrate MinimalTrainer etc
        # from src.trainer import setup_model_and_tokenizer, MinimalTrainer
        # model, tokenizer = setup_model_and_tokenizer(cfg)
        # trainer = MinimalTrainer(cfg, train_dataset, tokenizer)
        # trainer.train()
        # For now, just print stub
    elif args.command == "stats":
        cfg = load_train_config_from_yaml(args.config)
        print(f"Loaded config: {cfg}")
        # Try to load dataset and print quick stats
        from src.utils import load_jsonl, dataset_stats
        data = load_jsonl(cfg.dataset_path)
        stats = dataset_stats(data)
        print("Dataset stats:")
        if isinstance(stats, dict):
            import pprint
            pprint.pprint(stats)
        else:
            print(stats)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
