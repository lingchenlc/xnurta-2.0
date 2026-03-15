# Xnurta 2.0 算法升级行动计划

> 版本：v1.1 | 日期：2026-03-14
> 状态：Phase 1 完成，Phase 2 完成

---

## 总览

| 阶段 | 名称 | 目标 | 预计工期 | 状态 |
|------|------|------|----------|------|
| Phase 0 | 数据准备与探索 | 准备训练数据，理解数据分布 | 1-2 天 | ✅ 完成 |
| Phase 1 | AdTFT 时序预测模型 | 替代 AutoGluon，实现分位数预测 | 3-5 天 | ✅ 完成 |
| Phase 2 | 搜索词语义智能引擎 | LLM 意图分析 + Embedding 聚类 | 3-5 天 | ✅ 完成 |
| Phase 3 | Bid Landscape 预测 | 出价-流量响应曲线建模 | 3-4 天 | ⬜ 待开始 |
| Phase 4 | 集成与评估 | 三个模型联调，AB 测试框架 | 2-3 天 | ⬜ 待开始 |
| Phase 5 | Foundation Model | 跨账户预训练 + Per-account LoRA | 未来规划 | ⬜ 未来 |
| Phase 6 | RL Agent | 多智能体强化学习决策系统 | 未来规划 | ⬜ 未来 |

总计第一层（Phase 0-4）：约 12-19 个工作日

---

## Phase 0：数据准备与探索（1-2 天）

### 0.1 目标
- 拿到原始数据，理解字段含义
- 数据清洗、质量检查
- EDA（探索性分析），发现数据特点
- 确定建模策略

### 0.2 你需要准备的数据

#### 数据集 A：Campaign Daily Report（必需 ⭐⭐⭐）

| # | 字段名 | 类型 | 说明 | 必需 |
|---|--------|------|------|------|
| 1 | date | date | 日期，格式 YYYY-MM-DD | ✅ |
| 2 | account_id | string | 账户标识（可脱敏） | ✅ |
| 3 | marketplace | string | 站点：US/UK/DE/JP 等 | ✅ |
| 4 | campaign_id | string | Campaign 标识 | ✅ |
| 5 | campaign_name | string | Campaign 名称 | ✅ |
| 6 | campaign_type | string | SP/SB/SD | ✅ |
| 7 | campaign_status | string | enabled/paused/archived | ✅ |
| 8 | daily_budget | float | 当日设定的日预算 | ✅ |
| 9 | target_acos | float | 目标 ACoS（0-1 之间） | ✅ |
| 10 | ai_personality | int | AI 个性 1-5 | ✅ |
| 11 | impressions | int | 曝光数 | ✅ |
| 12 | clicks | int | 点击数 | ✅ |
| 13 | spend | float | 花费（美金） | ✅ |
| 14 | orders | int | 订单数 | ✅ |
| 15 | sales | float | 销售额（美金） | ✅ |

**数据量要求：**
- 最低：1 个账户，50+ Campaigns，90 天 → 约 4,500 行
- 理想：3-5 个账户，200+ Campaigns，180 天 → 约 100K-200K 行
- 最佳：10+ 账户，500+ Campaigns，360 天 → 约 1M+ 行

**文件格式：** CSV（UTF-8 编码），或 JSON Lines

---

#### 数据集 B：Targeting Daily Report（必需 ⭐⭐⭐）

| # | 字段名 | 类型 | 说明 | 必需 |
|---|--------|------|------|------|
| 1 | date | date | 日期 | ✅ |
| 2 | account_id | string | 账户标识 | ✅ |
| 3 | campaign_id | string | Campaign 标识 | ✅ |
| 4 | ad_group_id | string | 广告组标识 | ✅ |
| 5 | targeting_id | string | targeting 标识 | ✅ |
| 6 | targeting_type | string | keyword / product_targeting / auto | ✅ |
| 7 | targeting_text | string | 关键词文本或 ASIN | ✅ |
| 8 | match_type | string | exact/phrase/broad/auto | ✅ |
| 9 | bid | float | 当日出价 | ✅ |
| 10 | impressions | int | 曝光数 | ✅ |
| 11 | clicks | int | 点击数 | ✅ |
| 12 | spend | float | 花费 | ✅ |
| 13 | orders | int | 订单数 | ✅ |
| 14 | sales | float | 销售额 | ✅ |

**数据量要求：**
- 最低：1 个账户，500+ targeting，90 天 → 约 45K 行
- 理想：3-5 个账户，5000+ targeting，180 天 → 约 1M-5M 行

---

#### 数据集 C：Search Term Report（必需 ⭐⭐⭐）

| # | 字段名 | 类型 | 说明 | 必需 |
|---|--------|------|------|------|
| 1 | date | date | 日期（或汇总周期的起止日期） | ✅ |
| 2 | account_id | string | 账户标识 | ✅ |
| 3 | campaign_id | string | Campaign 标识 | ✅ |
| 4 | ad_group_id | string | 广告组标识 | ✅ |
| 5 | targeting_text | string | 匹配到的关键词 | ✅ |
| 6 | search_term | string | 用户实际搜索的词 | ✅ |
| 7 | impressions | int | 曝光数 | ✅ |
| 8 | clicks | int | 点击数 | ✅ |
| 9 | spend | float | 花费 | ✅ |
| 10 | orders | int | 订单数 | ✅ |
| 11 | sales | float | 销售额 | ✅ |

**数据量要求：**
- 最低：1 个账户，60 天 → 约 50K-200K 行
- 理想：3-5 个账户，180 天 → 约 1M-5M 行

---

#### 数据集 D：操作日志（非常有价值 ⭐⭐）

| # | 字段名 | 类型 | 说明 | 必需 |
|---|--------|------|------|------|
| 1 | timestamp | datetime | 操作时间 | ✅ |
| 2 | account_id | string | 账户 | ✅ |
| 3 | entity_type | string | campaign/ad_group/targeting | ✅ |
| 4 | entity_id | string | 被操作的实体 ID | ✅ |
| 5 | action_type | string | bid_change/budget_change/negate/harvest/pause/enable | ✅ |
| 6 | old_value | float/string | 操作前的值 | ✅ |
| 7 | new_value | float/string | 操作后的值 | ✅ |
| 8 | trigger_rule | string | 触发该操作的规则名称 | 可选 |
| 9 | remark | json | Xnurta 1.0 的 remark JSON | 可选 |

**说明：** 这是 Xnurta 1.0 每次执行优化时的操作记录。如果有，对 Bid Landscape 建模极其有价值。

**数据量要求：**
- 有多少给多少，越多越好
- 最低：30 天操作日志

---

#### 数据集 E：商品信息（有价值 ⭐）

| # | 字段名 | 类型 | 说明 | 必需 |
|---|--------|------|------|------|
| 1 | asin | string | 商品 ASIN | ✅ |
| 2 | product_title | string | 商品标题 | ✅ |
| 3 | product_category | string | 商品类目 | 可选 |
| 4 | product_price | float | 商品价格 | 可选 |
| 5 | account_id | string | 所属账户 | ✅ |

**说明：** 用于搜索词语义分析时判断关键词与商品的相关性。

---

#### 数据集 F：日内小时级数据（加分项 ⭐）

如果能提供小时粒度的 campaign/targeting 数据，对日内预算分配和出价优化非常有价值。
格式与数据集 A/B 相同，增加 `hour` 字段 (0-23)。

---

### 0.3 数据脱敏建议

```
account_id → hash 或编号即可（Account_001, Account_002...）
campaign_name → 可保留（含语义信息）或 hash
targeting_text/search_term → 必须保留原文（模型需要语义信息）
金额数据 → 保留原值（模型需要真实分布）
```

### 0.4 Phase 0 我会做的事

1. **数据加载与质量检查**
   - 缺失值统计
   - 异常值检测（负数、超大值、零值）
   - 时间连续性检查（是否有断天）

2. **探索性分析（EDA）**
   - 各指标的分布（直方图、箱线图）
   - 时序趋势可视化
   - Campaign/Targeting 层级的聚合统计
   - ACoS 分布 vs Target ACoS 分布
   - 数据稀疏性分析（多少 targeting 日数据 < 1 次点击）

3. **特征工程预研**
   - 计算派生指标：CTR, CVR, CPC, ROAS
   - 滚动窗口特征可行性（7d/14d/30d MA）
   - 季节性分解（STL 分解）

4. **输出建模计划**
   - 根据实际数据特点调整模型架构
   - 确定训练/验证/测试集划分方式
   - 确定评估指标

### 0.5 交付物
- `data_quality_report.html` — 数据质量报告
- `eda_notebook.ipynb` — 探索性分析 Notebook
- `modeling_plan.md` — 根据实际数据调整后的建模计划

---

## Phase 1：AdTFT 时序预测模型（3-5 天）

### 1.1 目标
构建一个 Temporal Fusion Transformer 变体，替代 AutoGluon，实现：
- 多步预测（1d / 3d / 7d / 14d）
- 分位数预测（P10 / P25 / P50 / P75 / P90）
- 可解释的特征重要性和时间注意力权重
- 与 AI Personality 1-5 的原生集成

### 1.2 使用的数据
- 数据集 A（Campaign Daily Report）
- 数据集 B（Targeting Daily Report）
- 数据集 F（小时级数据，如果有的话）

### 1.3 具体步骤

#### Day 1：数据预处理 + 特征工程
```
输入：原始 CSV
处理：
  1. 时间对齐（补充缺失日期，填充零值）
  2. 计算派生特征：
     - CTR = clicks / impressions
     - CVR = orders / clicks
     - CPC = spend / clicks
     - ACoS = spend / sales
     - ROAS = sales / spend
  3. 滚动窗口特征：
     - 7d / 14d / 30d 移动平均
     - 7d / 14d 标准差（波动率）
     - 环比变化率
  4. 日历特征：
     - day_of_week (0-6)
     - is_weekend (bool)
     - month (1-12)
     - Fourier 季节性编码 (sin/cos pairs, periods=[7, 30, 365])
  5. 促销日标记：
     - Prime Day, Black Friday, Cyber Monday, Christmas...
     - 各站点特有促销日
  6. 标签构造：
     - 未来 1d/3d/7d/14d 的 ACoS, Sales, Spend, Orders
输出：PyTorch Dataset 类
```

#### Day 2：模型架构实现
```python
# 实现以下模块：
1. VariableSelectionNetwork — 自动特征选择
2. GatedResidualNetwork — 非线性特征变换
3. TemporalSelfAttention — 可解释的时间注意力
4. QuantileOutput — 分位数回归输出层

# 模型配置（初始版本）
config = {
    'hidden_size': 256,
    'attention_heads': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 1,
    'dropout': 0.1,
    'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
    'prediction_horizons': [1, 3, 7, 14],
    'lookback_window': 60,  # 用过去 60 天预测
    'static_features': ['campaign_type', 'match_type', 'marketplace', 'ai_personality'],
    'known_future_features': ['day_of_week', 'is_weekend', 'is_promo', 'fourier_*'],
    'observed_features': ['impressions', 'clicks', 'spend', 'orders', 'sales',
                          'ctr', 'cvr', 'cpc', 'acos', 'roas',
                          'ma_7d_*', 'ma_14d_*', 'std_7d_*'],
}
```

#### Day 3：训练 Pipeline
```
1. 数据划分：
   - 训练集：前 70% 时间段
   - 验证集：接下来 15% 时间段
   - 测试集：最后 15% 时间段
   （时序数据必须按时间切分，不能随机切分）

2. 训练配置：
   - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
   - Scheduler: OneCycleLR or CosineAnnealing
   - Loss: Quantile Loss (pinball loss)
   - Batch Size: 64-128
   - Epochs: 50-100 with early stopping (patience=10)
   - Mixed Precision: FP16

3. 训练监控：
   - 验证集 Quantile Loss
   - 各预测步长的 MAPE / RMSE / MAE
   - 覆盖率检验：P10-P90 区间是否真的覆盖 80% 的真实值
```

#### Day 4：评估与调优
```
1. 定量评估：
   - MAPE / RMSE / MAE（对比 AutoGluon baseline）
   - 分位数校准曲线（Calibration Plot）
   - 各 AI Personality 的风险-收益分析

2. 定性评估：
   - 时间注意力可视化（模型关注哪些历史时间步）
   - 特征重要性排名（Variable Selection 权重）
   - 极端情况分析（促销日前后、新 Campaign 冷启动）

3. 调优：
   - 超参数搜索（hidden_size, n_heads, lookback_window）
   - 数据增强（时序 jitter, 随机 mask）
   - 集成学习（多模型 bagging 取平均分位数）
```

#### Day 5：封装与部署准备
```
1. 推理接口封装：
   class AdTFTPredictor:
       def predict(self, campaign_id, targeting_id=None) -> PredictionResult
       # PredictionResult 包含各预测步长的分位数

2. AI Personality 集成：
   class PersonalityAwareDecision:
       def get_decision_point(self, prediction, personality) -> float
       # 根据 Personality 选择对应分位数

3. 模型序列化 + 版本管理

4. 与 Xnurta 1.0 的接口适配文档
```

### 1.4 交付物
- `models/ad_tft/` — 模型代码
- `models/ad_tft/trained/` — 训练好的模型权重
- `evaluation_report.html` — 评估报告（含可视化）
- `api_interface.py` — 推理接口
- `comparison_vs_autogluon.md` — 与 AutoGluon 的对比报告

---

## Phase 2：搜索词语义智能引擎（3-5 天）

### 2.1 目标
- 搜索词自动分类（品牌/竞品/泛流量/精准/无关）
- 语义聚类（10 万词 → 300 个簇）
- LLM 驱动的否定安全审查
- 关键词机会发现

### 2.2 使用的数据
- 数据集 C（Search Term Report）
- 数据集 E（商品信息）

### 2.3 具体步骤

#### Day 1：Embedding 基建
```
1. 选择并部署 Embedding 模型：
   - 方案 A：BGE-M3（多语言，本地部署）
   - 方案 B：E5-large-v2（英文为主场景）
   - 方案 C：调用 OpenAI/Anthropic Embedding API（最简单）

2. 对所有搜索词和关键词计算 embedding
   - 批量处理，结果存储为 .npy 或 FAISS index
   - 建立 term → embedding 的快速检索

3. 构建 FAISS 向量索引
   - 支持快速最近邻搜索
   - 用于后续聚类和相似度查询
```

#### Day 2：搜索词聚类
```
1. HDBSCAN 聚类
   - 在 embedding 空间做密度聚类
   - 自动确定簇数，不需要预设 K

2. 簇级别指标聚合
   - 每个簇的总 impressions / clicks / orders / spend / sales
   - 簇级 ACoS, CVR, CTR

3. 簇标签生成
   - 方案 A：取每个簇中心最近的 5 个词作为标签
   - 方案 B：调用 LLM 为每个簇生成语义标签

4. 可视化
   - t-SNE / UMAP 降维可视化
   - 簇表现热力图
```

#### Day 3：LLM 意图分析 Pipeline
```
1. 设计 Prompt 模板
   - 搜索词意图分类 prompt
   - 否定安全审查 prompt
   - 关键词建议 prompt

2. 构建批量处理框架
   - LLM API 调用管理（并发、限速、重试）
   - 结果解析与校验
   - 缓存层（避免重复调用）

3. 成本优化
   - 轻量模型预过滤（embedding 相似度 < 0.3 直接标记无关）
   - LLM 只处理需要深度分析的词
   - batch prompt（一次分析多个词）
```

#### Day 4：否定词安全引擎
```
1. 替代 Sentence Transformer 方案
   - 多维度安全评分：
     a. Embedding 相似度（快速筛选层）
     b. LLM 语义判断（深度分析层）
     c. 历史数据交叉验证（如果该词在其他 Campaign 有转化，不否定）

2. 安全等级分类：
   - 绿灯：可安全否定
   - 黄灯：需要人工确认
   - 红灯：禁止否定

3. 对比测试：
   - 用历史否定记录做回测
   - 统计误否定率 vs 1.0 系统
```

#### Day 5：关键词机会发现 + 集成
```
1. 关键词拓展
   - LLM 生成候选长尾词
   - Embedding 最近邻搜索扩展
   - 按预期 ACoS 和搜索量评分排名

2. 整合输出
   - 统一接口：输入 Search Term Report → 输出完整分析报告
   - 报告包含：分类、聚类、否定建议、拓展建议

3. 可视化 Dashboard 数据
   - 导出聚类结果为可交互的 HTML 报告
```

### 2.4 交付物
- `models/semantic_engine/` — 语义引擎代码
- `models/semantic_engine/embeddings/` — 预计算的 embedding
- `analysis_report.html` — 搜索词分析报告
- `negation_backtest.md` — 否定词回测报告

---

## Phase 3：Bid Landscape 预测（3-4 天）

### 3.1 目标
- 给定 targeting，预测不同出价下的曝光/点击/转化
- 生成出价-效果响应曲线
- 自动找到满足 Target ACoS 约束的最优出价

### 3.2 使用的数据
- 数据集 B（Targeting Daily Report，含 bid 字段）
- 数据集 D（操作日志 — 出价变动记录）
- Phase 1 的 AdTFT 预测结果

### 3.3 前置条件
**关键：** 这个模型的质量高度依赖于 **出价变动数据的丰富程度**。
- 如果每个 targeting 只有 1 个固定出价，无法建模响应曲线
- 需要大量"出价变了 → 表现怎么变"的观测对
- Xnurta 1.0 的操作日志是最有价值的数据源

如果操作日志不够丰富，这个 Phase 可以延后到积累足够数据后再做。

### 3.4 具体步骤

#### Day 1：因果数据构造
```
1. 从操作日志提取"出价干预事件"
   - 找到所有 bid_change 事件
   - 记录变动前 7 天和变动后 7 天的表现
   - 构造 (before_bid, after_bid, before_metrics, after_metrics) 对

2. 混杂因素识别
   - 季节性、竞争环境变化可能同时影响出价和表现
   - 使用 day_of_week, trend 等作为控制变量

3. 数据量统计
   - 如果有效出价变动事件 < 1000 个，考虑降级为简单模型
   - 如果 > 10000 个，可以训练完整的 Landscape 模型
```

#### Day 2-3：模型构建
```
方案 A（数据充足时）：单调神经网络
  - 约束 bid ↑ → impressions ↑（单调递增）
  - 输入：targeting_embedding + context + bid
  - 输出：predicted_impressions, predicted_clicks, predicted_orders

方案 B（数据不足时）：参数化曲线拟合
  - 假设 impressions = f(bid) 服从 log-linear 关系
  - 每个 targeting 拟合独立参数
  - 简单但实用

方案 C（混合方案）：
  - 用 A 或 B 做粗预测
  - 用 AdTFT 的分位数预测做修正
```

#### Day 4：最优出价求解器 + 封装
```
1. 给定 Bid Landscape + Target ACoS + AI Personality
   → 自动求解最优出价

2. 可视化：生成出价曲线图
   → 标注当前出价位置、最优区间、危险区域

3. 推理接口封装
```

### 3.5 交付物
- `models/bid_landscape/` — 模型代码
- `bid_curve_examples.html` — 出价曲线示例可视化
- `optimal_bid_report.md` — 最优出价分析报告

---

## Phase 4：集成与评估（2-3 天）

### 4.1 目标
- 三个模型联调，确保数据流通畅
- 与 Xnurta 1.0 系统的接口对接
- 构建 AB 测试框架
- 完整的端到端评估

### 4.2 具体步骤

#### Day 1：统一接口 + 联调
```
1. 构建 Xnurta2Pipeline 类
   - 输入：account_data
   - 流程：
     a. AdTFT 预测各 targeting 未来表现
     b. Bid Landscape 找最优出价
     c. 语义引擎分析搜索词，给出投放/否定建议
   - 输出：优化建议列表（与 1.0 格式兼容）

2. 与 1.0 系统对接
   - 输出格式适配
   - 回退机制：如果 2.0 模型异常，自动降级到 1.0
```

#### Day 2：AB 测试框架
```
1. 设计 AB 分流机制
   - Campaign 级别分流（不是 targeting 级别）
   - 控制组：Xnurta 1.0 决策
   - 实验组：Xnurta 2.0 决策
   - 建议比例：70% 控制 / 30% 实验（初期保守）

2. 监控看板
   - 实时对比两组的 ACoS, Sales, Spend, ROAS
   - 统计显著性检验（每日更新 p-value）
   - 安全护栏：如果实验组 ACoS 恶化超过 X%，自动停止
```

#### Day 3：评估报告 + 文档
```
1. 完整评估报告
   - 各模型的离线指标
   - 预期收益估算
   - 风险评估

2. 技术文档
   - 模型架构文档
   - 接口文档
   - 运维手册（模型更新、监控、故障排查）
```

### 4.3 交付物
- `pipeline/` — 集成 pipeline 代码
- `ab_test_framework/` — AB 测试框架
- `docs/` — 完整技术文档
- `final_evaluation.html` — 最终评估报告

---

## Phase 5 & 6：未来规划（概要）

### Phase 5：Foundation Model（预计 1-2 个月）

**额外数据需求：**
- 10+ 账户的完整数据（跨品类、跨站点）
- 至少 6 个月历史
- 数据量级：千万行

**产出：**
- 预训练的 Ad Foundation Model
- Per-account LoRA 适配能力
- 新账户 7 天冷启动能力

### Phase 6：RL Agent（预计 2-3 个月）

**额外数据需求：**
- Xnurta 1.0 的全量操作日志（构建 Offline RL 数据集）
- 实时数据管道（Kafka + Flink）
- 环境模拟器校准数据

**产出：**
- Budget / Bid / Targeting / Structure 四个 RL Agent
- 多智能体协同训练框架
- 全自动优化能力

---

## 数据提交检查清单

在开始 Phase 0 之前，请按优先级准备数据：

- [ ] ⭐⭐⭐ 数据集 A：Campaign Daily Report（CSV）
- [ ] ⭐⭐⭐ 数据集 B：Targeting Daily Report（CSV）
- [ ] ⭐⭐⭐ 数据集 C：Search Term Report（CSV）
- [ ] ⭐⭐ 数据集 D：操作日志（CSV/JSON）
- [ ] ⭐ 数据集 E：商品信息（CSV）
- [ ] ⭐ 数据集 F：小时级数据（CSV，如果有）

**文件放置位置：**
```
./data/raw/
├── campaign_daily.csv
├── targeting_daily.csv
├── search_term_report.csv
├── operation_logs.csv (如果有)
├── product_info.csv (如果有)
└── hourly_data.csv (如果有)
```

---

## 项目文件结构

```
./
├── ACTION_PLAN.md          ← 本文档
├── data/
│   ├── raw/                ← 原始数据放这里
│   ├── processed/          ← 处理后的数据
│   └── features/           ← 特征数据
├── notebooks/
│   ├── 00_eda.ipynb        ← Phase 0 探索分析
│   ├── 01_adtft.ipynb      ← Phase 1 模型实验
│   ├── 02_semantic.ipynb   ← Phase 2 语义分析
│   └── 03_landscape.ipynb  ← Phase 3 出价曲线
├── models/
│   ├── ad_tft/             ← 时序预测模型
│   ├── semantic_engine/    ← 语义引擎
│   └── bid_landscape/      ← 出价预测
├── pipeline/               ← Phase 4 集成
├── ab_test/                ← AB 测试框架
├── docs/                   ← 文档
└── utils/                  ← 公共工具函数
```
