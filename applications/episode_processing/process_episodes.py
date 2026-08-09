#!/usr/bin/env python3
"""
处理VLN episodes数据
将各城市的VLN_episodes.json按照城市和类型整理成train和eval数据集

Usage:
    python process_episodes.py ver_260202
    python process_episodes.py ver_260202 --train-only
    python process_episodes.py ver_260202 --eval-only
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    from .config import (
        TRAIN_CITIES,
        EPISODE_TYPES,
        EPISODE_FILES,
        get_data_dir,
        get_episodes_dir,
        classify_eval_cities,
    )
except ImportError:
    from config import (
        TRAIN_CITIES,
        EPISODE_TYPES,
        EPISODE_FILES,
        get_data_dir,
        get_episodes_dir,
        classify_eval_cities,
    )


def load_episodes(city_path: Path) -> List[Dict[str, Any]]:
    """加载指定城市的episodes"""
    file_path = city_path / "VLN_episodes.json"
    if not file_path.exists():
        print(f"  Warning: {file_path} not found, skipping...")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("episodes", [])


def filter_episodes_by_type(episodes: List[Dict], trajectory_type: str) -> List[Dict]:
    """按trajectory_type筛选episodes"""
    return [ep for ep in episodes if ep.get("trajectory_type") == trajectory_type]


def save_episodes(episodes: List[Dict], output_path: Path):
    """保存episodes到json文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=4, ensure_ascii=False)
    print(f"  Saved {len(episodes)} episodes to {output_path}")


def process_cities(data_dir: Path, cities: List[str], output_dir: Path, split_name: str):
    """处理一组城市的数据"""
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name.upper()} data")
    print(f"Cities: {', '.join(cities)}")
    print(f"{'=' * 60}")

    all_episodes = []

    # 加载所有城市的数据
    for city in cities:
        city_path = data_dir / city
        if not city_path.exists():
            print(f"  Warning: City directory {city_path} not found, skipping...")
            continue

        print(f"Loading {city}...")
        episodes = load_episodes(city_path)
        all_episodes.extend(episodes)
        print(f"  - {city}: {len(episodes)} episodes")

    if not all_episodes:
        print(f"  No episodes found for {split_name}")
        return

    # 按类型分类
    boundary_episodes = filter_episodes_by_type(all_episodes, EPISODE_TYPES["boundary"])
    landmark_episodes = filter_episodes_by_type(all_episodes, EPISODE_TYPES["landmark"])
    road_episodes = filter_episodes_by_type(all_episodes, EPISODE_TYPES["road"])

    # 统计信息
    print(f"\nTotal episodes: {len(all_episodes)}")
    print(f"  - Boundary: {len(boundary_episodes)}")
    print(f"  - LandmarkSet: {len(landmark_episodes)}")
    print(f"  - Road: {len(road_episodes)}")

    # 保存文件
    split_output_dir = output_dir / split_name
    save_episodes(all_episodes, split_output_dir / EPISODE_FILES["all"])
    save_episodes(boundary_episodes, split_output_dir / EPISODE_FILES["boundary"])
    save_episodes(landmark_episodes, split_output_dir / EPISODE_FILES["landmark"])
    save_episodes(road_episodes, split_output_dir / EPISODE_FILES["road"])

    return {
        "total": len(all_episodes),
        "boundary": len(boundary_episodes),
        "landmark": len(landmark_episodes),
        "road": len(road_episodes)
    }


def process_episodes(
    version: str,
    train_only: bool = False,
    eval_only: bool = False,
):
    """
    处理指定版本的episodes数据

    Args:
        version: 数据集版本名称 (e.g., ver_260202)
        train_only: 只处理训练集
        eval_only: 只处理评估集
    """
    data_dir = get_data_dir(version)
    output_dir = get_episodes_dir(version)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Processing episodes for version: {version}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    results = {}

    if not eval_only:
        results["train"] = process_cities(data_dir, TRAIN_CITIES, output_dir, "train")

    if not train_only:
        results.update(_process_eval_splits(data_dir, output_dir))

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"{'=' * 60}")

    return results


def _process_eval_splits(data_dir: Path, output_dir: Path) -> dict:
    """在 eval/ 下生成 val_seen/ 和 val_unseen/ 子目录。

    seen/unseen 按城市名前缀自动判断：eval 城市的基础名若在 train 中有任意
    TIF，则为 val_seen；否则为 val_unseen。
    """
    val_seen_cities, val_unseen_cities = classify_eval_cities()

    print(f"\n{'=' * 60}")
    print("Classifying eval cities into val_seen / val_unseen")
    print(f"  val_seen  ({len(val_seen_cities)}): {', '.join(val_seen_cities) or '(none)'}")
    print(f"  val_unseen({len(val_unseen_cities)}): {', '.join(val_unseen_cities) or '(none)'}")
    print(f"{'=' * 60}")

    results = {}
    if val_seen_cities:
        results["val_seen"] = process_cities(
            data_dir, val_seen_cities, output_dir / "eval", "val_seen"
        )
    else:
        print("  val_seen: no cities, skipping.")
        results["val_seen"] = None

    if val_unseen_cities:
        results["val_unseen"] = process_cities(
            data_dir, val_unseen_cities, output_dir / "eval", "val_unseen"
        )
    else:
        print("  val_unseen: no cities, skipping.")
        results["val_unseen"] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="处理VLN episodes数据，按城市和类型整理成train/eval数据集"
    )
    parser.add_argument(
        "version",
        type=str,
        help="数据集版本名称 (e.g., ver_260202)"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="只处理训练集"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只处理评估集"
    )

    args = parser.parse_args()

    if args.train_only and args.eval_only:
        parser.error("Cannot specify both --train-only and --eval-only")

    process_episodes(
        args.version,
        train_only=args.train_only,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
