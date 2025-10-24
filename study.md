MLLM4Text-ReID-main/
├── model/               # 模型定义目录
│   ├── build.py         # 预训练模型构建
│   ├── build_finetune.py # 微调模型构建
│   ├── clip_model.py    # 基于CLIP的基础模型
│   └── objectives.py    # 损失函数定义
├── datasets/            # 数据集相关
│   ├── build.py         # 数据加载器构建
│   └── bases.py         # 基础数据集类
├── processor/           # 训练和评估处理器
│   ├── processor.py     # 预训练处理器
│   └── processor_finetune.py # 微调处理器
├── solver/              # 优化器和学习率调度器
│   ├── build.py         # 优化器和调度器构建
│   └── lr_scheduler.py  # 学习率调度器
├── utils/               # 工具函数
│   └── options.py       # 命令行参数设置
├── train.py             # 预训练入口
├── finetune.py          # 微调入口
├── run.sh               # 预训练执行脚本
└── finetune.sh          # 微调执行脚本

模型构建 (build.py 和 build_finetune.py)

- IRRA类 ：主要模型类，负责特征提取和融合
  - 初始化方法：设置任务、构建基础模型、定义分类器和跨模态Transformer
  - cross_former方法：实现交叉注意力和Transformer处理
  - encode_image/encode_text方法：分别提取图像和文本特征
  - forward方法：根据任务不同处理输入并返回特征或损失

基础模型 (clip_model.py)
- Transformer类 ：实现了带有modal参数的Transformer编码器
  - forward方法需要x和modal两个参数，根据modal类型处理不同模态的数据
  - 包含多层ResidualAttentionBlock来处理序列特征

项目提供多种损失函数以支持不同的训练目标，主要定义在objectives.py中：

- SDM损失 (compute_sdm) ：相似分布匹配损失
  - 计算图像-文本特征之间的相似性分布
  - 使用KL散度匹配预测分布和真实分布
  - 考虑相同ID和图像ID的约束

- MLM损失 (compute_mlm) ：掩码语言模型损失
  - 用于训练模型理解文本语义

- ID损失 (compute_id) ：实例分类损失
  - 使模型学习区分不同行人的能力

- ITC损失 (compute_itc) ：图像-文本对比损失
  - 基于InfoNCE的对比学习损失

- CMPM损失 (compute_cmpm) ：跨模态投影匹配损失
  - 优化跨模态投影的匹配性能

- Patch损失 (compute_patch) ：补丁级别的匹配损失
  - 处理图像补丁特征和文本特征的匹配

### 3. 数据处理模块
数据处理相关代码位于datasets目录：

- 数据集工厂 (build.py) ：根据名称创建不同的数据集实例
  
  - 支持CUHK-PEDES、ICFG-PEDES、RSTPReid等数据集
  - 定义图像变换和批处理函数
- 基础数据集类 (bases.py) ：
  
  - BaseDataset：所有数据集的基类
  - ImageTextDataset：处理图像和文本对的数据集
  - ImageDataset：仅处理图像的数据集
  - 实现了文本tokenization和图像预处理功能
