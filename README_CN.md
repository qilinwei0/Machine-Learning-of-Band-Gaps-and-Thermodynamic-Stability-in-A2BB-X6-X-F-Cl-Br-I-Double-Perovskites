# `final_complete_corrected_package_v9/data` 机器学习复现代码

本代码包根据原始任务 CSV 生成 `data/raw` 和 `data/tables` 的全部机器学习数据。读取以 `feat_` 开头的398个特征，并以现有 `split_formula_group` 作为唯一的训练、验证和测试划分依据。

## 1. 输入文件

把以下三个文件放在同一文件夹：

- `A2BBX6_task_bandgap_regression.csv`
- `A2BBX6_task_hull_regression.csv`
- `A2BBX6_task_stability_classification_005.csv`
## 2. 安装依赖

建议使用 Python 3.10–3.13：

```bash
python -m pip install -r requirements.txt
```

## 3. 完整运行

Windows：

```bat
run_v9_ml.bat "D:\你的数据文件夹" "D:\输出\v9"
```

Linux/macOS：

```bash
bash run_v9_ml.sh "/path/to/input" "/path/to/output/v9"
```

也可直接运行：

```bash
python generate_v9_ml_data.py --input-dir /path/to/input --output-root /path/to/v9
```

完整模式默认包括：500棵 Extra Trees、HistGradientBoosting、3次×4折重复分组交叉验证、500次 Bootstrap、10次置换重要性。CPU不同会影响运行时间，但相同 Python/NumPy/pandas/scikit-learn 版本和随机种子下结果可复现。

首次检查可用快速模式：

```bash
python generate_v9_ml_data.py --input-dir /path/to/input --output-root /path/to/test_output --quick
```

快速模式只用于验证环境与输出格式，不能替代论文正式结果。

## 4. 方法设计

- 数据划分：严格复用 `split_formula_group`，禁止按行随机拆分，防止同一化学式的多晶型跨集合泄漏。
- 输入特征：仅使用 `feat_` 列；`mpid`、元素符号、空间群、质量控制字段和 `target_` 标签不进入模型。
- 预处理：缺失值中位数插补只在训练集或交叉验证训练折内拟合；Ridge和逻辑回归的标准化也只在训练层拟合。
- 模型比较：回归比较均值基线、Ridge、Extra Trees、HistGradientBoosting；分类比较多数类基线、逻辑回归、Extra Trees、HistGradientBoosting。
- 固定主模型：带隙回归采用 Extra Trees；凸包能回归和稳定性分类采用 HistGradientBoosting。
- 分类阈值：仅在固定验证集上以 MCC 选择，测试集不参与阈值选择。
- 重复交叉验证：仅在训练集+验证集组成的开发集内做公式分组交叉验证；固定测试集始终封存。
- Bootstrap：对固定测试集成对重采样，生成95%置信区间。
- 保形预测：固定验证集作为 calibration set，测试集只用于覆盖率评价。
- 特征消融：全特征、非结构特征、仅 `feat_struct_` 结构特征三组。
- 候选筛选：预测带隙0.8–2.5 eV、预测凸包能不高于0.05 eV/atom、预测稳定概率不低于0.5；综合分数只用于组内排序。

## 5. 输出

程序生成：

- `data/raw/`：三份任务输入的原样副本；
- `data/tables/`：18张 CSV 表；
- `data/tables/ml_run_metadata.json`：运行环境、随机种子及参数记录。

生成后执行：

```bash
python validate_v9_data.py --package-root /path/to/v9
```

校验程序检查18张表的存在性、列数、空表以及三份原始任务数据的公式分组交叉泄漏。

## 6. 重要说明

模型结果会受到 scikit-learn 版本和并行计算环境的轻微影响。若要逐数值复现某次历史运行，请保留该次生成的 `ml_run_metadata.json` 并使用相同依赖版本。

