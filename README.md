# PyCaret-tutorial
PyCaret练练手

### 安装  

PyCaret 2.0版本：
```bash
pip install pycaret==2.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install shap -i https://mirrors.aliyun.com/pypi/simple/  # 图形可视化
brew install libomp  # 如果macos上遇到lightgbm报错则执行该安装
```

brew安装：
```bash
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

PyCaret 2.3版本(建议安装Python 3.8)：
```bash
# pip install pycaret==2.3 -i https://mirrors.aliyun.com/pypi/simple/
pip install pycaret[full] -i https://mirrors.aliyun.com/pypi/simple/ # 安装包括XGBoost的所有模型
```

### 教程

PyCaret文档地址: https://pycaret.readthedocs.io/en/latest/index.html


### 运行
* 建议在Jupyter Notebook上运行, 方便结果查看, PyCharm Professional有较好的支持;

### 结果
* 分类：查看src/tutorial_0_classification.ipynb



