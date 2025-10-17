# 训练监控器 - 快速入门指南

## 🚀 5分钟快速开始

### 第一步：安装依赖

```bash
# 进入项目目录
cd /Users/HarowrdLiu/learn/robot/kuavo_data_challenge

# 安装基础依赖（必需）
pip install tensorboard

# 安装高级功能依赖（推荐）
pip install rich matplotlib psutil GPUtil
```

### 第二步：测试安装

```bash
# 运行测试脚本验证安装
python kuavo_train/test_monitor.py
```

### 第三步：开始监控

```bash
# 方式1：使用快捷脚本（推荐）
./kuavo_train/monitor.sh

# 方式2：直接运行Python脚本
python kuavo_train/monitor_training.py
```

---

## 📱 常用命令速查

### 基础监控

```bash
# 自动找到最新训练并显示状态
./kuavo_train/monitor.sh

# 自动刷新（每5秒更新一次）
python kuavo_train/monitor_training.py --refresh 5

# 生成训练报告和图表
python kuavo_train/monitor_training.py --save-report --plot
```

### 高级监控

```bash
# 启动实时仪表板
./kuavo_train/monitor.sh advanced

# 启动实时图表
./kuavo_train/monitor.sh plot

# 启用GPU监控
./kuavo_train/monitor.sh gpu
```

### 监控指定训练

```bash
# 监控特定训练运行
python kuavo_train/monitor_training.py --run-dir outputs/train/task1_moving_grasp/smolvla_sequential/run_20251017_120000
```

---

## 🎯 实际使用场景

### 场景1：训练开始后，快速检查是否正常

```bash
# 启动训练后，等待5-10分钟，然后运行：
./kuavo_train/monitor.sh

# 查看输出：
# - "训练Loss: 0.XXXX 📉 (-X%)" → Loss在下降，正常
# - "状态: 正常下降" → 训练健康
# - 如果看到警告，根据建议调整
```

### 场景2：长时间训练，持续监控

```bash
# 启动高级监控器，自动刷新
./kuavo_train/monitor.sh gpu

# 或者在另一个终端窗口运行：
python kuavo_train/monitor_training.py --refresh 10

# 让它在后台运行，定期看一眼即可
```

### 场景3：训练过程中发现loss不下降

```bash
# 生成详细报告和图表
python kuavo_train/monitor_training.py --save-report --plot

# 查看loss曲线，诊断问题：
# - Loss平稳不动 → 学习率可能太小
# - Loss震荡很大 → 学习率可能太大
# - Loss突然上升 → 可能遇到bad batch或需要调整
```

### 场景4：对比不同超参数的训练效果

```bash
# 分别查看两次训练的曲线
python kuavo_train/monitor_training.py --run-dir outputs/train/.../run1 --plot
python kuavo_train/monitor_training.py --run-dir outputs/train/.../run2 --plot

# 对比loss下降速度和最终收敛值
```

---

## 📊 关键指标解读

### Loss趋势
- **📉 正常**: Loss持续下降 → 继续训练
- **📊 震荡**: Loss上下波动 → 考虑降低学习率
- **📈 上升**: Loss增大 → 检查学习率或数据

### 学习率
- **最佳范围**: 1e-5 ~ 1e-3
- **过小** (<1e-7): 训练太慢
- **过大** (>1e-2): 不稳定

### 健康度评分
- **90-100分**: 优秀，继续保持
- **70-89分**: 良好，小问题
- **50-69分**: 需要调整
- **<50分**: 严重问题，立即处理

---

## 🔧 故障排除

### 问题：找不到训练数据

```bash
# 检查训练是否正在运行
ps aux | grep train

# 检查输出目录
ls -la outputs/train/
```

### 问题：依赖缺失

```bash
# 一键安装所有依赖
./kuavo_train/monitor.sh install-deps
```

### 问题：GPU监控不工作

```bash
# 检查NVIDIA驱动
nvidia-smi

# 安装GPUtil
pip install gputil
```

---

## 💡 专业技巧

### 技巧1：在训练机器上运行监控

```bash
# SSH到训练机器
ssh user@training-machine

# 启动监控（终端模式）
cd /path/to/project
./kuavo_train/monitor.sh advanced

# 或者用tmux/screen后台运行
tmux new -s monitor
./kuavo_train/monitor.sh gpu
# Ctrl+B, D 分离会话
```

### 技巧2：定期保存报告

```bash
# 每小时自动保存一次报告
while true; do
    python kuavo_train/monitor_training.py --save-report
    sleep 3600
done
```

### 技巧3：使用别名简化命令

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias monitor='cd /path/to/project && ./kuavo_train/monitor.sh'
alias monitor-gpu='cd /path/to/project && ./kuavo_train/monitor.sh gpu'
alias monitor-plot='cd /path/to/project && ./kuavo_train/monitor.sh plot'

# 然后可以直接运行
monitor        # 快速监控
monitor-gpu    # GPU监控
monitor-plot   # 查看图表
```

### 技巧4：结合TensorBoard使用

```bash
# 终端1: 启动TensorBoard
tensorboard --logdir outputs/train --port 6006

# 终端2: 启动监控器
./kuavo_train/monitor.sh advanced

# 浏览器: http://localhost:6006
# 可以同时看到详细的TensorBoard界面和终端监控
```

---

## 📚 完整文档

详细文档请查看：[TRAINING_MONITOR_README.md](./TRAINING_MONITOR_README.md)

---

## 🆘 需要帮助？

1. **查看完整文档**: `cat kuavo_train/TRAINING_MONITOR_README.md`
2. **运行测试**: `python kuavo_train/test_monitor.py`
3. **查看帮助**: `./kuavo_train/monitor.sh help`

---

## ✅ 检查清单

在开始使用前，请确认：

- [ ] 已安装 Python 3.7+
- [ ] 已安装 tensorboard（必需）
- [ ] 已安装 rich, matplotlib（推荐）
- [ ] 已有训练运行（outputs/train/下有数据）
- [ ] 脚本有执行权限（chmod +x）

全部打勾后，就可以开始使用了！🎉

