# MonoDepth-RS

## 📥 下载预训练模型

**模型地址**：  
[百度网盘链接](https://pan.baidu.com/s/1H41V78ddq6KIT3BD60jLeg)  
**提取码**：`sn5a`

---

## 🧱 Installation

建议使用 Conda 环境管理：

```bash
# 创建并激活 Conda 环境
conda create -n iebins python=3.8
conda activate iebins
```

```bash
# 安装 PyTorch（根据你的 GPU 匹配版本）
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch
```

```bash
# 安装项目依赖
pip install matplotlib tqdm tensorboardX timm mmcv open3d
```

---

## 🏋️‍♂️ Training

训练 WHU-OMVS 模型：

```bash
python iebins/whu_train.py uav_configs/arguments_train_whu.txt
```

训练 WHU-MVS 模型：

```bash
python iebins/whu_mvs_train.py uav_configs/arguments_train_whu.txt
```

---

## 📊 Evaluation

评估 WHU-OMVS 模型：

```bash
python iebins/whu_eval.py uav_configs/arguments_eval_whu.txt
```

评估 WHU-MVS 模型（基于 SUN RGB-D 数据集）：

```bash
python iebins/whu_mvs_eval.py uav_configs/arguments_eval_whu_mvs.txt
```

---

## 🖼️ Inference

生成 WHU-OMVS 推理图像：

```bash
python iebins_kittiofficial/test.py uav_configs/arguments_test_whu.txt
```

---

## 📁 项目结构（可选）

```text
MonoDepth-RS/
├── iebins/
│   ├── whu_train.py
│   ├── whu_eval.py
│   └── whu_mvs_train.py
├── iebins_kittiofficial/
│   └── test.py
├── uav_configs/
│   ├── arguments_train_whu.txt
│   ├── arguments_eval_whu.txt
│   └── arguments_test_whu.txt
└── README.md
```
