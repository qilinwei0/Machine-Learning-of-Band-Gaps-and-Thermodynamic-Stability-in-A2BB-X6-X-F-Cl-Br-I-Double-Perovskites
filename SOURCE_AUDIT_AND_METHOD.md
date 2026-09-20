# 源文件核查与代码设计依据

## 已核查的文件关系

1. `Supplementary Data S1_A2BCX6_semiconductor_with_features.xls`
   - 1个工作表，2490行、129列。
   - 含材料标识、化学式、空间群、能带类型、形成能、凸包能、结构文本、元素分数及原始聚合特征。
2. `A2BBX6_phase4_feature_engineering(1).py`
   - 从富集 CSV 与元素属性表生成以 `feat_` 开头的组成、位点、B/B′交换不变、物理和结构特征。
   - 生成固定的 `split_formula_group`，并把标签映射到 `target_` 命名空间。
3. `A2BBX6_phase4_prepare_ml(1).py`
   - 演示单一目标的训练集内中位数插补、可选标准化及 NumPy 矩阵导出。
   - 该脚本不能生成 v9 的模型比较、预测、消融、交叉验证、Bootstrap、保形预测和候选筛选表。
4. `A2BBX6_final_ml_dataset_report.xlsx`
   - 明确主数据含398个结构增强特征。
   - 明确要求仅使用 `feat_` 列、复用 `split_formula_group`、所有预处理只在训练层拟合。
5. `final_complete_corrected_package_v9.zip/data/raw`
   - 实际只包含带隙回归、凸包能回归、稳定性分类三份任务 CSV。
6. `final_complete_corrected_package_v9.zip/data/tables`
   - 共18张 CSV，覆盖模型比较、测试预测、数据质量、消融、特征重要性、重复分组交叉验证、Bootstrap、保形预测及候选筛选。

## 新代码解决的问题

- 把原来的单目标矩阵准备扩展成三任务、端到端、可复现的机器学习流水线。
- 所有模型统一从任务 CSV 的398个 `feat_` 特征读取数据。
- 在程序入口处检查固定划分完整性，并禁止公式组跨训练、验证、测试集合。
- 将中位数插补和线性模型标准化封装进 scikit-learn `Pipeline`，确保每次交叉验证都只在训练折拟合。
- 稳定性分类的阈值只由固定验证集确定；测试集不参与阈值、特征或超参数选择。
- 保留固定测试集，仅在训练集+验证集组成的开发集内进行重复公式分组交叉验证。
- 自动生成与旧v9完全一致的18张 CSV 文件名和列结构，并额外记录运行环境。

## 标签处理说明

旧数据准备脚本默认目标为 `target_is_gap_direct_existing`，而最终数据报告明确要求直接带隙分类使用更新后的 `target_is_gap_direct_mp`。本v9机器学习代码不训练直接带隙分类，因为该任务不在v9 `data/raw` 的三条主线中，从而不会误用旧标签。若后续扩展直接带隙模型，应单独读取 `A2BBX6_task_direct_gap_classification.csv` 并指定 `target_is_gap_direct_mp`。

