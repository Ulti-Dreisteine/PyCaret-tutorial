# PyCaret-tutorial
PyCaret练练手

### 安装  

PyCaret 2.0版本 (不建议安装)：
```bash
pip install pycaret==2.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install shap -i https://mirrors.aliyun.com/pypi/simple/  # 图形可视化
brew install libomp  # 如果macos上遇到lightgbm报错则执行该安装
```

brew安装：
```bash
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

PyCaret 2.3版本 (建议安装, 同时对应Python 3.8版本)：
```bash
# pip install pycaret==2.3 -i https://mirrors.aliyun.com/pypi/simple/
pip install pycaret -i https://mirrors.aliyun.com/pypi/simple/
pip install pycaret[full] -i https://mirrors.aliyun.com/pypi/simple/ # 安装包括XGBoost的所有模型
```

如果出现

```
ImportError: Missing optional dependency 'Jinja2' . DataFrame.style requires jinja2. Use pip or conda to install Jinja2.
```

则升级markupsafe包:

```
pip install markupsafe==2.0.1
```

### 教程

PyCaret文档地址: https://pycaret.readthedocs.io/en/latest/index.html


### 运行
* 建议在Jupyter Notebook上运行, 方便结果查看, PyCharm Professional有较好的支持;

### 结果
* 分类：查看src/tutorial_0_classification.ipynb



