#!/usr/bin/env python3
"""
合并指定目录下所有 episode JSON 文件

用法:
    python merge_episodes.py <input_dir> [--output <output_file>] [--reassign-ids]

示例:
    python merge_episodes.py /path/to/episodes/train
    python merge_episodes.py /path/to/episodes/train --output merged.json
    python merge_episodes.py /path/to/episodes/train --reassign-ids
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any


def find_episode_files(input_dir: str) -> List[Path]:
    """查找目录下所有的 episode JSON 文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"目录不存在: {input_dir}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"不是有效目录: {input_dir}")
    
    # 查找所有 JSON 文件
    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"目录下没有找到 JSON 文件: {input_dir}")
    
    return json_files


def load_episodes(json_file: Path) -> List[Dict[str, Any]]:
    """从 JSON 文件加载 episodes"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'episodes' in data:
        return data['episodes']
    elif isinstance(data, list):
        # 如果文件直接是 episodes 列表
        return data
    else:
        print(f"警告: {json_file} 格式不符合预期，跳过")
        return []


def merge_episodes(
    input_dir: str,
    output_file: str = None,
    reassign_ids: bool = False
) -> Dict[str, Any]:
    """
    合并目录下所有 episode JSON 文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径，如果为 None 则默认为 input_dir/merged_episodes.json
        reassign_ids: 是否重新分配 episode_id（从 0 开始连续编号）
    
    Returns:
        合并后的数据
    """
    # 查找所有 JSON 文件
    json_files = find_episode_files(input_dir)
    print(f"找到 {len(json_files)} 个 JSON 文件:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # 合并所有 episodes
    all_episodes = []
    file_stats = []
    
    for json_file in json_files:
        episodes = load_episodes(json_file)
        count = len(episodes)
        all_episodes.extend(episodes)
        file_stats.append((json_file.name, count))
        print(f"加载 {json_file.name}: {count} 个 episodes")
    
    print(f"\n总共合并 {len(all_episodes)} 个 episodes")
    
    # 重新分配 episode_id（如果需要）
    if reassign_ids:
        print("重新分配 episode_id...")
        for i, episode in enumerate(all_episodes):
            episode['episode_id'] = i
    
    # 构建输出数据
    merged_data = {
        "episodes": all_episodes
    }
    
    # 确定输出文件路径
    if output_file is None:
        output_file = os.path.join(input_dir, "merged_episodes.json")
    
    # 写入文件
    print(f"\n写入合并结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    # 打印统计信息
    print("\n=== 合并统计 ===")
    print(f"{'文件名':<40} {'Episode 数量':>15}")
    print("-" * 60)
    for name, count in file_stats:
        print(f"{name:<40} {count:>15}")
    print("-" * 60)
    print(f"{'总计':<40} {len(all_episodes):>15}")
    
    return merged_data


def main():
    parser = argparse.ArgumentParser(
        description="合并目录下所有 episode JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python merge_episodes.py /path/to/episodes/train
    python merge_episodes.py /path/to/episodes/train --output merged.json
    python merge_episodes.py /path/to/episodes/train --reassign-ids
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="包含 episode JSON 文件的目录路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径 (默认: <input_dir>/merged_episodes.json)"
    )
    
    parser.add_argument(
        "--reassign-ids",
        action="store_true",
        help="重新分配 episode_id，从 0 开始连续编号"
    )
    
    args = parser.parse_args()
    
    try:
        merge_episodes(
            input_dir=args.input_dir,
            output_file=args.output,
            reassign_ids=args.reassign_ids
        )
        print("\n合并完成!")
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
