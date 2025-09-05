# kuavo_train

`kuavo_train` 提供了一个基于 **LeRobot Diffusion Policy** 的封装，支持深度图像处理、Transformer 结构以及多模态信息融合。通过该封装，用户可以方便地在自定义任务中快速构建和训练自己的策略模型。

## 特性

- **Diffusion Policy 封装**：封装了训练、配置及推理流程，方便直接使用。
- **Transformer 支持**：策略网络支持 Transformer 结构，提升时序特征建模能力。
- **深度图像处理与融合**：可处理多摄像头深度图像，支持特征融合。
- **易于扩展**：其他策略可以参考现有封装，通过继承或修改进行定制。

## 文件结构

```
kuavo_train/
├── utils/
│ ├── __init__.py
│ ├── augmenter.py # 数据增广工具
│ ├── transforms.py # 基于lerobot的数据增广，包含transformconfigs
│ └── utils.py # 其他辅助函数，如保存和载入随机数状态
├── wrapper/
│ ├── dataset/
│ │ └── LeRobotDatasetWrapper.py # 数据集封装，仅提供了数据集继承示例，目前实际未使用
│ ├── policy/
│ │ └── diffusion/
│ │ ├── __init__.py
│ │ ├── DiffusionConfigWrapper.py   # diffusion config策略配置继承示例
│ │ ├── DiffusionModelWrapper.py    # diffusion model模型继承示例，包含深度图像、特征融合等
│ │ ├── DiffusionPolicyWrapper.py   # diffusion policy策略继承示例，包含裁剪crop，缩放resize等输入处理
│ │ ├── DiT_1D_model.py             # 基于DiT魔改的一维数据扩散模型，可选
│ │ ├── DiT_model.py                # 基于DiT的扩散模型，可选
│ │ └── transformer_diffusion.py    # 基于transformer的diffusion扩散模型
│ └── __init__.py
├── README.md
└── train_policy.py # 策略训练入口
```

## 使用说明

1. **准备数据集**  
   使用 `python kuavo_data/CvtRosbag2Lerobot.py` 转换数据集。

2. **配置策略**  
   在 `configs/policy/diffusion_config.yaml` 中设置训练、模型、策略的参数。

3. **训练策略**  
   通过 `python kuavo_train/train_policy.py` 启动训练。

4. **扩展其他策略**  
   - 可参考 `DiffusionPolicyWrapper.py` 的继承结构，实现自定义策略。
   - 若策略需要处理深度图像或融合多模态信息，可参考`DiffusionModelWrapper.py`
   - 扩散模型可复用现有 Transformer 的模块。

## 依赖

- PyTorch
- torchvision
- 其他依赖请参考 `requirements_ilcode.txt`等, 以及项目整体[README.md](../README.md)

---

> 本目录主要面向机器人模仿学习策略学习研究，可作为快速构建 diffusion-based policy 的模板。