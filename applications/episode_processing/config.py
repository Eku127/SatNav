#!/usr/bin/env python3
"""
数据处理配置文件
"""

import os
from pathlib import Path
from typing import List, Tuple

# 数据集根目录。机器路径通过 SatNav 自己的环境变量注入。
DATASET_ROOT = Path(os.environ.get("SATNAV_DATA_ROOT", "data/satnav")).expanduser()

# 城市分类配置（0327 默认划分）
# eval: LosAngeles-1, Rome-1, NewYork-1, Auckland-1, Orlando-1, Rotterdam-1
# train: 其余全部城市（含 Amsterdam-1, Dube-1）
TRAIN_CITIES = [
    "Amsterdam-1",
    "Amsterdam-2",
    "Berlin-1",
    "Berlin-2",
    "Berlin-3",
    "Berlin-4",
    "Berlin-5",
    "Boston-1",
    "Boston-2",
    "Boston-3",
    "Boston-4",
    "Boston-5",
    "Brugge-1",
    "Dube-1",
    "Geneva-1",
    "Geneva-2",
    "Geneva-3",
    "Geneva-4",
    "Geneva-5",
    "London-1",
    "London-2",
    "London-3",
    "London-4",
    "LosAngeles-2",
    "LosAngeles-3",
    "LosAngeles-4",
    "Minneapolis-1",
    "Minneapolis-2",
    "Minneapolis-3",
    "Minneapolis-4",
    "Minneapolis-5",
    "Minneapolis-6",
    "NewYork-2",
    "NewYork-3",
    "NewYork-4",
    "NewYork-5",
    "Paris-1",
    "Paris-2",
    "Paris-3",
    "Paris-4",
    "Paris-5",
    "RiodeJaneiro-1",
    "Rome-2",
    "Rome-3",
    "Rome-4",
    "Rome-5",
    "Sydney-1",
    "TheBayArea-1",
    "TheBayArea-2",
    "TheBayArea-3",
    "TheBayArea-4",
    "TheBayArea-5",
    "TheBayArea-6",
]
EVAL_CITIES = [
    "LosAngeles-1",
    "Rome-1",
    "NewYork-1",
    "Auckland-1",
    "Orlando-1",
    "Rotterdam-1",
]
ALL_CITIES = TRAIN_CITIES + EVAL_CITIES


def classify_eval_cities(
    eval_cities: List[str] = None,
    train_cities: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """将 eval 城市按 seen/unseen 分类。

    规则：eval 城市的基础城市名（如 "LosAngeles-1" → "LosAngeles"）
    若在 train_cities 中有任意 TIF 出现，则归入 val_seen；否则归入 val_unseen。

    Returns:
        (val_seen_cities, val_unseen_cities)
    """
    if eval_cities is None:
        eval_cities = EVAL_CITIES
    if train_cities is None:
        train_cities = TRAIN_CITIES

    train_bases = {city.rsplit("-", 1)[0] for city in train_cities}

    val_seen, val_unseen = [], []
    for city in eval_cities:
        base = city.rsplit("-", 1)[0]
        if base in train_bases:
            val_seen.append(city)
        else:
            val_unseen.append(city)

    return val_seen, val_unseen

# Episode 类型
EPISODE_TYPES = {
    "boundary": "Boundary",
    "landmark": "LandmarkSet",
    "road": "Road"
}

# 输出文件名
EPISODE_FILES = {
    "all": "all_episodes.json",
    "boundary": "boundary_episodes.json",
    "landmark": "landmark_episodes.json",
    "road": "road_episodes.json"
}

def get_dataset_path(version: str) -> Path:
    """获取数据集版本路径"""
    return DATASET_ROOT / version


def get_data_dir(version: str) -> Path:
    """获取数据目录路径"""
    return get_dataset_path(version) / "data"


def get_episodes_dir(version: str) -> Path:
    """获取episodes输出目录"""
    return get_dataset_path(version) / "episodes"
