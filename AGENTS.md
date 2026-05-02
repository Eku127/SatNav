# AGENTS.md

## Startup Context (Mandatory)

- **每次启动一个新的 agent 时，必须先阅读：**
  - `README.md`（项目总览、环境安装、核心概念）
- 若 `.codex/CODEX_CONTEXT.md` 存在，优先以该文件作为当前仓库的事实来源（路径、脚本、环境、约定）。

## Context Maintenance Rule (Mandatory)

- 若本次任务包含**重大改动**，必须在结束前同步更新 `README.md`（或 `.codex/CODEX_CONTEXT.md` 如已创建）。
- 重大改动包括但不限于：
  - 目录/包结构重构
  - 仿真器接口或任务配置变化
  - 数据格式、episode 结构变化
  - 关键脚本路径或环境约定变化

## Repo Overview

SatNav 是一个连续状态 VLN（视觉语言导航）评测平台，使用卫星地图作为场景：

- **仿真器架构**：`SatSim`（2D 卫星图）
- **核心包**：`satnav/`（含 `core/`、`dataset/`、`navigation/`、`sims/`、`task/`、`training/`、`models/`）
- **配置文件**：`configs/`（`default.yaml`、`satnav_task.yaml`、`configs/baselines/`）
- **脚本**：`scripts/`
- **应用示例**：`applications/`

## Repo Skills

- If task matches seq2seq training (训练seq2seq, 启动seq2seq训练, run seq2seq training), use:
  - `.codex/skills/seq2seq-train/SKILL.md`
- If task matches seq2seq evaluation (评测seq2seq, eval seq2seq, 查看seq2seq结果, run seq2seq eval), use:
  - `.codex/skills/seq2seq-eval/SKILL.md`
- If task matches CMA training (训练cma, 启动cma训练, run cma training), use:
  - `.codex/skills/cma-train/SKILL.md`
- If task matches CMA evaluation (评测cma, eval cma, 查看cma结果, run cma eval), use:
  - `.codex/skills/cma-eval/SKILL.md`
- If task matches SatNav baseline smoke test (冒烟测试, smoke test, quick validation), use:
  - `.codex/skills/baseline-smoke-test/SKILL.md`

## Execution Rules

- 优先使用 `configs/` 下现有配置文件，通过 `run.py` 启动任务。
- 修改仿真器参数时，优先编辑 `configs/satnav_task.yaml`，避免硬编码。
- 新增 baseline 配置请放入 `configs/baselines/` 目录。
- 数据集 episode 相关逻辑位于 `satnav/dataset/`，导航逻辑位于 `satnav/navigation/`。
- 当前仓库不保留自动化测试目录；验证优先使用 quickstart 和模型训练/评测脚本。
