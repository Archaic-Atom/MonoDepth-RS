# MonoDepth-RS

**📥 下载预训练模型**  
[点此下载模型（百度网盘）](https://pan.baidu.com/s/1H41V78ddq6KIT3BD60jLeg)  
提取码：`sn5a`

<div align="center">

---

## 🔧 Installation

建议使用 Conda 虚拟环境进行安装：

```bash
# 创建并激活 Conda 环境
conda create -n MonoRS python=3.8
conda activate MonoRs

# 安装 PyTorch（根据你的显卡选择合适的 cudatoolkit）
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch

# 安装依赖包
pip install matplotlib tqdm tensorboardX timm mmcv open3d
