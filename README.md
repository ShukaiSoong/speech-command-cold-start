# speech-command-cold-start
Speech command classification system for intelligent hardware cold-start scenarios.
# Speech Command Cold Start System

基于讯飞生态的轻量化算法实习项目，面向智能硬件语音指令的快速冷启动系统。

## 项目简介

针对智能硬件场景中新设备、新方言数据不足的问题，构建轻量化语音指令分类系统，完成数据处理、模型训练、迁移学习、量化部署及实时Demo开发。

## 主要工作

* 复现官方Baseline（ResNet152）
* MobileNetV3轻量化实验
* SpecAugment数据增强
* 粤语Few-shot迁移实验
* INT8量化部署
* Streamlit实时演示系统

## 实验结果

| 模型                             | Macro-F1 |
| ------------------------------ | -------- |
| ResNet152                      | 0.8435   |
| MobileNetV3Large               | 0.7538   |
| MobileNetV3Large + SpecAugment | 0.8335   |

部署结果：

| 模型   | Accuracy |
| ---- | -------- |
| FP32 | 0.8750   |
| INT8 | 0.8708   |

## 项目结构

```text
├── data
├── checkpoints
├── src
├── demo
├── reports
└── README.md
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行

训练：

```bash
python src/train.py
```

Demo：

```bash
streamlit run demo/app.py
```

## 项目亮点

* 超过官方Baseline性能（0.8435 > 0.829）
* 模型压缩至16.22MB
* 完成跨语种迁移实验
* 完成INT8量化部署
* 支持实时语音交互演示
