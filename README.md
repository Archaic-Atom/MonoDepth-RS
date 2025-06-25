# MonoDepth-RS

## 📥 下载已训练的对比模型

**模型地址**：  
[百度网盘链接](https://pan.baidu.com/s/1H41V78ddq6KIT3BD60jLeg)  
**提取码**：`sn5a`

---

## 🧱 Installation

建议使用 Conda 环境管理：

```bash
conda create -n MonoRSpython=3.8
conda activate MonoRS
```

```bash
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch
```

```bash
pip install matplotlib tqdm tensorboardX timm mmcv open3d
```

---

## 🏋️‍♂️ Training

训练模型：


```bash
python MonoRS/anything_train.py uav_configs/arguments_train_anything.txt
```

---

## 📊 Evaluation

评估模型：

```bash
python MonoRS/whu_eval.py uav_configs/arguments_eval_whu.txt
```

---

## 🖼️ Inference

生成推理图像：

```bash
python MonoRS/anything_test.py uav_configs/arguments_test_whu.txt
```

---

