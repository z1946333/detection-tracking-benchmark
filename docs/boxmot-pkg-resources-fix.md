# BoxMOT 导入错误修复记录

## 现象

在 `yahui_torch` 环境执行 `import boxmot` 时出现：

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

## 原因

当前环境的 `setuptools 82.0.1` 已不再提供 `pkg_resources`，而 BoxMOT 10.0.75 的导入链仍依赖该兼容模块。

## 修复

在 VS Code 的 PowerShell 终端执行：

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' -m pip install setuptools==80.9.0
```

验证：

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' -c "import pkg_resources; print('pkg_resources OK')"
& 'D:\conda_envs\yahui_torch\python.exe' -c "from boxmot import StrongSORT, HybridSORT; print('BoxMOT trackers OK')"
```

## 当前脚本适配

`scripts/run_boxmot.py` 使用 BoxMOT 10.0.74/10.0.75 的类名：

- `StrongSORT`
- `HybridSORT`

GPU 参数使用 `device="0"`。这里的 `"0"` 表示第一张 NVIDIA GPU；不要写 `device="cuda"`，否则 BoxMOT 会把字符串 `cuda` 当成设备编号。

## 结果解释

这属于 BoxMOT 版本适配运行，不应在报告中写成原论文官方实现的严格复现。应记录 BoxMOT 版本、ReID 权重和本项目的检测器权重。
