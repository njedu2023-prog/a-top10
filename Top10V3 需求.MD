# a-top10 V3 需求文件

## 0. 文档定位

本文档作为 `njedu2023-prog/a-top10` 后续 V3 升级的**正式需求锚点 / 开发主线 / 验收主文档**。

目标不是讨论抽象方案，而是为后续开发提供：

* 统一目标
* 统一口径
* 统一字段契约
* 统一阶段推进顺序
* 统一验收标准
* 统一失败回退与安全升级机制

本文档优先级高于临时讨论结论。后续开发若与本文档冲突，应先修正文档，再执行代码修改。

---

## 1. V3 最高目标

### 1.1 唯一核心目标

**在 T 日收盘后，预测 T+1 最大概率涨停的股票，并输出 Top10。**

系统最高价值不是“能跑通”，也不是“学习着玩”，而是：

> **持续逼近“次日最大概率涨停标的”的真实分布，并在 Top10 资源位中尽可能提高命中率与可交易价值。**

### 1.2 非目标

以下不是 V3 的主目标：

* 只做展示好看
* 只做报告完整
* 只做工程跑通
* 只做简单规则打分
* 只追求模型文件更新
* 只追求样本数量增长

### 1.3 V3 成功定义

V3 成功不是单点成功，而是以下五条同时成立：

1. **上游数据契约稳定**
2. **FeatureHistory 成为高质量学习特征池**
3. **Step5 概率链稳定产出，不再大量空值/镜像污染**
4. **Step7 训练更新链可控、可解释、可回滚**
5. **Top10 输出在真实交易日持续具备预测价值**

---

## 2. 当前问题总诊断（V2 → V3 的起点）

### 2.1 当前已具备

当前系统已经具备以下基础：

* 主链可跑
* 输出链可跑
* 学习链已接通
* `replay / train / auto_daily` 运行模式契约已初步建立
* `prob_rule / prob_ml / prob_final / final_score` 主字段口径已引入
* `feature_history.csv` 已存在，学习样本池已开始累积
* `step7_report_latest.json` 已能输出训练判定与质量闸门状态

### 2.2 当前核心短板

当前系统距离“机构级涨停预测系统”仍然明显偏粗，核心原因不是代码数量不够，而是以下结构性短板：

#### （1）FeatureHistory 信息密度不足

当前 `outputs/learning/feature_history.csv` 仍偏薄，主要问题：

* 原生市场特征偏少
* 板质量字段偏少
* 市场状态字段缺失
* 行为历史字段缺失
* 题材生态字段过粗
* 概率镜像字段占比偏高

结果是：

> 样本“可训练”，但只支持粗学习，不支持精学习。

#### （2）质量闸门已立住，但被关键字段覆盖率卡住

当前系统不是没有训练框架，而是：

* `train_rows` 已经足够
* `level` 已经达到高等级
* 但 `feature_coverage` 不足
* `prob_final` 等关键字段覆盖质量不够稳定
* `final_score` 存在退化风险

结果是：

> 系统处于“会跑，但不稳定成长”的状态。

#### （3）模型层尚未真正成为主驾驶

虽然 Step5 已引入 ML 概率框架，但从系统级看，仍是：

* 规则骨架偏重
* 学习肌肉偏弱
* 排序逻辑仍偏工程拼接

#### （4）输出可看，但验收不够残酷

系统有报告，但当前验收仍不足以支撑机构级迭代，缺：

* 关键字段覆盖率分解
* 各阶段失败可解释性
* 分层命中率
* 场景化回测/评估
* 模型更新安全发布机制

#### （5）升级节奏尚未形成“安全有序”的系统纪律

当前存在风险：

* 一改字段，容易牵动主链
* 一改学习，容易污染正式模型
* 一改输出，容易影响下游消费
* 缺少 staging → release 的安全升级轨道

---

## 3. V3 总体原则

### 3.1 第一原则：以预测价值为唯一最高标准

所有升级、字段扩容、模型训练、报告改动，都必须服从于：

> **能否提高 T+1 涨停预测能力。**

### 3.2 第二原则：先打厚特征，再谈更花模型

V3 的主线不是先追新模型，而是先把：

* 上游原始字段
* 学习样本密度
* 质量闸门
* 标签与评估

打到足够专业。

### 3.3 第三原则：安全升级，不污染正式模型

所有训练更新都必须具备：

* 门槛判定
* staging 保存
* 验收后发布
* 失败可回退

### 3.4 第四原则：学习层与结果层分离

* `feature_history.csv` 可以很厚
* 最终结果表不应无脑膨胀
* 对外产物结构尽量稳定

### 3.5 第五原则：先契约，后开发

后续开发不允许“想到哪改到哪”。

必须先明确：

* 字段英文名
* 来源表
* 主键
* 落地逻辑
* 缺失策略
* 闸门阈值
* 验收方式

---

## 4. V3 顶层系统架构

V3 保持 Step2 ~ Step7 主链框架，但内部升级为以下结构：

### 4.1 数据层

* 上游 raw 快照层
* 标准化特征抽取层
* 学习样本池（FeatureHistory）
* 历史预测池
* 历史真值池

### 4.2 推理层

* Step2：候选池生成
* Step3：板强/行为/市场特征抽取
* Step4：题材与热度特征抽取
* Step5：规则概率 + ML 概率 + 融合概率
* Step6：最终排序 / Top10 输出

### 4.3 学习层

* 样本成熟判断
* 质量闸门
* 训练集构造
* LR / LGBM 训练
* staging 模型发布
* 正式模型替换

### 4.4 验收层

* 字段覆盖率报告
* 样本质量报告
* 模型更新报告
* 命中率与结构化回顾

---

## 5. 上游数据契约（V3 正式版）

## 5.1 上游源表清单

V3 以 `a-top10/_warehouse/a-share-top3-data/data/raw/YYYY/YYYYMMDD/` 为标准上游目录。

V3 固定使用以下表作为主源：

1. `_meta.json`
2. `daily.csv`
3. `daily_basic.csv`
4. `hot_boards.csv`
5. `limit_break_d.csv`
6. `limit_list_d.csv`
7. `limit_up_tags.csv`
8. `moneyflow_hsgt.csv`
9. `namechange.csv`
10. `stk_limit.csv`
11. `stock_basic.csv`
12. `top_list.csv`

## 5.2 主键契约

### 个股日级主键

* `trade_date`
* `ts_code`

### 题材日级主键

* `trade_date`
* `industry` 或 `board`

### 市场日级主键

* `trade_date`

## 5.3 预测时点可得性契约

所有进入 `feature_history.csv` 的特征必须满足：

> **在 T 日收盘后生成预测时已经可得。**

严禁以下字段直接进入 T 日特征：

* T+1 开盘价
* T+1 收盘价
* T+1 是否涨停
* T+1 是否可买入
* T+2 任何价格

这些只能作为：

* 标签
* 验证字段
* 回测字段

## 5.4 数据稳定性契约

新字段入池前必须满足以下最小标准：

1. 有明确来源表
2. 可按主键稳定对齐
3. 历史期内大部分交易日可获得
4. 单位与口径稳定
5. 不属于未来信息

---

## 6. FeatureHistory 升级主线（V3 核心）

## 6.1 FeatureHistory 定位

`outputs/learning/feature_history.csv` 是 V3 的**学习样本总池 / 特征仓**。

其职责不是展示，而是：

* 沉淀每个交易日、每只股票的预测时点特征
* 为 Step7 提供正式训练输入
* 为质量闸门提供字段级观测基础
* 为后续回测/诊断提供特征追踪依据

## 6.2 FeatureHistory 字段分层

V3 将字段分为四层：

### A. 主键与追踪字段

必须存在：

* `run_time_utc`
* `trade_date`
* `ts_code`
* `name`
* `run_mode`
* `run_id`（新增）
* `run_attempt`（新增）
* `commit_sha`（新增）
* `_meta_status`（新增）

### B. 当前核心基础字段（保留并清洗）

* `StrengthScore`
* `ThemeBoost`
* `seal_amount`
* `open_times`
* `turnover_rate`
* `Probability`
* `_prob_src`
* `_prob_warn`
* `prob_rule`
* `prob_ml`
* `prob_final`
* `prob_src`
* `prob_ml_available`
* `prob_fusion_mode`

### C. V3 必须新增字段（直接来源明确，可优先落地）

#### 价格/成交基础

* `open`
* `high`
* `low`
* `close`
* `pct_chg`
* `vol`
* `amount`

#### 日频基础属性

* `turnover_rate_f`
* `volume_ratio`
* `total_mv`
* `float_mv`

#### 板质量特征

* `up_limit`
* `down_limit`
* `limit_type`
* `fd_amount`
* `first_time`
* `last_time`
* `seal_amount`
* `open_times`

#### 题材/热度特征

* `industry`
* `is_hot_board`
* `board_rank`
* `board_limit_up_count`

#### 标的基础属性

* `market`
* `list_date`

#### 资金类特征

* `toplist_amount`
* `toplist_net_amount`
* `toplist_net_rate`
* `toplist_amount_rate`
* `toplist_float_values`
* `toplist_reason`

#### 市场整体状态特征

* `north_money`
* `south_money`
* `hgt`
* `ggt_ss`
* `ggt_sz`

### D. V3 应新增衍生字段（第二阶段）

#### 板质量衍生

* `seal_amount_ratio = seal_amount / amount`
* `fd_amount_ratio = fd_amount / amount`
* `limit_close_dist = (up_limit - close) / close`
* `is_first_limit = open_times == 0`
* `is_multi_open = open_times >= 2`

#### 题材生态衍生

* `industry_limitup_density`
* `is_top_hot_board`
* `board_hot_bucket`

#### 行为历史衍生

* `ret_1d`
* `ret_3d`
* `ret_5d`
* `amt_ratio_3d`
* `turnover_ratio_3d`
* `board_count_5d`
* `prev_limit_hit`

#### 标的属性衍生

* `price_bucket`
* `float_mv_bucket`
* `list_age_days`

---

## 7. FeatureHistory 字段入池映射表规范

V3 所有新增字段必须按以下映射表规范落地：

| 字段英文名       | 来源表                          | 主键                 | 落地逻辑                 | 缺失策略    | 闸门阈值       | 是否主特征 |
| ----------- | ---------------------------- | ------------------ | -------------------- | ------- | ---------- | ----- |
| seal_amount | limit_list_d / limit_break_d | trade_date+ts_code | 直接读取，优先 limit_list_d | 缺失置空并记录 | 非空率 ≥ 0.90 | 是     |

后续正式开发时，每个字段必须有一行映射，不允许“临时口头记住”。

---

## 8. Step2 ~ Step6 主链升级要求

## 8.1 Step2 候选池

V3 要求 Step2 不只是“筛股票”，还要为后续学习提供更完整候选空间。

### 要求

* 保留当前可运行主逻辑
* 输出候选池规模信息
* 输出候选池过滤原因统计
* 不允许因为过滤过严导致样本学习层过窄

### 验收

* 候选池数目可观测
* 各过滤条件有统计
* 候选池不是固定极小规模

## 8.2 Step3 强度特征

V3 要求 Step3 从“一个总分”升级为“总分 + 原始板质量特征 + 可解释衍生项”。

### 要求

* `StrengthScore` 保留
* 但不能只保留总分，必须同时把构成强度的关键原始字段落入 FeatureHistory

### 验收

* StrengthScore 可追溯
* 与 `open_times / seal_amount / first_time / last_time / amount / turnover_rate` 联动可诊断

## 8.3 Step4 题材特征

V3 要求 Step4 从单值 `ThemeBoost` 升级为“主题热度结构化特征”。

### 要求

* `ThemeBoost` 保留
* 同时补充：行业 / 热板标记 / 热度排名 / 同板涨停家数

### 验收

* ThemeBoost 不再是唯一题材表示
* 能从样本表反推出该票题材热度背景

## 8.4 Step5 概率链

V3 要求 Step5 成为**主驾驶之一**，不是只当“附加概率”。

### 要求

* `prob_rule` 与 `prob_ml` 语义清晰
* `prob_final` 必须稳定产生
* `prob_src` 与 `prob_fusion_mode` 必须可解释
* 不允许大量空值
* 不允许主字段退化为镜像垃圾字段集合

### 验收

* `prob_final` 非空率达标
* `prob_ml_available` 正确反映模型可用性
* `prob_final` 与 `prob_rule / prob_ml` 关系可解释

## 8.5 Step6 最终排序

V3 要求 Step6 保持对外口径稳定，但内部必须更可诊断。

### 要求

* `final_score` 作为唯一排序主字段
* `rank` 由 `final_score` 派生
* 保证 `final_score` 非常数列
* 输出 Top10 结果表继续控制字段数量，不做学习表式膨胀

### 验收

* `final_score` 非零率 > 0
* 标准差 > 0
* 唯一值数 > 1

---

## 9. 标签体系（V3）

## 9.1 主标签

V3 主标签仍为：

* `y_limit_up_t1`：T+1 是否涨停

这是当前最高优先级标签，必须稳定构造。

## 9.2 次级标签（V3.1 / V3.2 预备）

后续预留：

* `y_tradeable_t1`：T+1 是否具备可参与性
* `y_touch_limit_t1`：T+1 盘中是否触板
* `y_close_strong_t1`：T+1 收盘是否保持强势

### 原则

V3 主线先把主标签体系打稳，不在第一阶段无限扩展标签复杂度。

---

## 10. Step7 学习闭环升级

## 10.1 Step7 定位

Step7 不是“顺手训练一下”，而是：

> **样本成熟判断 + 质量闸门 + 正式训练 + 安全发布 + 可解释报告**

## 10.2 样本成熟度契约（沿用并收紧）

保留 Level 1 / 2 / 3 结构，但 V3 要求：

* Level 判定只是“有资格训练”
* 不等于“允许发布正式模型”

## 10.3 质量闸门重构为两层

### 第一层：结构闸门

重点检查主特征列是否存在且覆盖率达标。

至少检查：

* `StrengthScore`
* `ThemeBoost`
* `turnover_rate`
* `seal_amount`
* `open_times`
* `amount`
* `float_mv`
* `board_limit_up_count`

### 第二层：产出闸门

重点检查推理输出字段是否健康。

至少检查：

* `prob_final`
* `prob_ml`
* `final_score`
* `prob_fusion_mode`

## 10.4 闸门失败输出要求

失败时必须明确输出：

* 哪一层失败
* 哪个字段失败
* 失败指标值
* 阈值
* 是否允许 fallback

例如：

* `skip_train: struct_gate_fail: seal_amount_missing_rate=0.23 > 0.10`
* `skip_train: output_gate_fail: prob_final_missing_rate=0.18 > 0.05`
* `skip_train: output_gate_fail: final_score_std=0.0`

---

## 11. 模型发布与安全升级机制

## 11.1 模型文件角色

以下文件是**正式模型产物**，不是每次运行必重写的缓存：

* `models/step5_lr.joblib`
* `models/step5_lgbm.joblib`

## 11.2 staging 发布轨道（V3 新增）

训练成功后，先保存到：

* `models/staging/step5_lr.joblib`
* `models/staging/step5_lgbm.joblib`

并生成：

* `models/staging/model_meta.json`

其中至少记录：

* `run_id`
* `trade_date`
* `train_rows`
* `pos`
* `feature_coverage`
* `commit_sha`
* `run_mode`
* `quality_gate_pass`

## 11.3 正式发布条件

只有在 staging 满足发布条件时，才覆盖正式模型：

* 训练成功
* 结构闸门通过
* 产出闸门通过
* 模型文件保存成功
* 报告落盘成功

## 11.4 回退机制

若任一条件失败：

* 保持正式模型不变
* staging 可保留供检查
* `updated = false`
* `reason` 必须写明

---

## 12. 输出层契约（学习层与结果层分离）

## 12.1 学习层文件

V3 重点文件：

* `outputs/learning/feature_history.csv`
* `outputs/learning/pred_top10_history.csv`
* `outputs/learning/step7_report_latest.json`
* `outputs/learning/step7_report_latest.md`
* `outputs/learning/step7_hit_rate_history.csv`

### 原则

这些文件可以随 V3 升级扩充观测字段。

## 12.2 结果层文件

对外结果表尽量稳定：

* `outputs/learning/pred_top10_latest.csv`
* `outputs/learning/pred_top10_YYYYMMDD.csv`
* `outputs/predict_top10_YYYYMMDD.json`
* `outputs/predict_top10_YYYYMMDD.md`
* `outputs/latest.md`

### 原则

* 不应无脑同步全部新特征
* 只增加少量解释性最强字段
* 优先保持对外消费稳定

## 12.3 调试层文件

V3 鼓励增加调试观测，但不污染结果层。

例如：

* 字段覆盖率摘要
* 闸门失败摘要
* 特征分布摘要
* staging 发布摘要

---

## 13. 验收体系（V3）

## 13.1 上游数据验收

每个交易日要能回答：

* 12 张表是否都存在
* 各表 rows 数是否合理
* 是否存在空表
* 是否影响主特征构造

## 13.2 FeatureHistory 验收

必须验收：

* 行数增长是否合理
* 关键字段覆盖率
* 新增字段非空率
* 主字段分布是否退化

## 13.3 概率链验收

必须验收：

* `prob_rule` 是否存在
* `prob_ml` 是否存在
* `prob_final` 是否稳定
* `final_score` 是否非常数
* `rank` 是否与 `final_score` 一致

## 13.4 模型更新验收

必须同时验证：

* `trained`
* `updated`
* 模型文件 commit 是否变化
* staging 是否落盘
* 正式模型是否替换

## 13.5 预测价值验收

至少逐步建立：

* Top1 命中率
* Top3 命中率
* Top5 命中率
* Top10 命中率
* 按市场状态分桶命中率

---

## 14. 开发主线与阶段推进

## 14.1 V3 主线总顺序

后续开发严格按以下主线推进：

### 第一阶段：上游字段契约盘清

目标：逐张验清 12 张源表
产物：字段入池映射表 V1

### 第二阶段：FeatureHistory 扩容

目标：把必须字段安全落池
产物：升级后的 `feature_history.csv`

### 第三阶段：质量闸门重构

目标：结构闸门 + 产出闸门可解释
产物：新的 `step7_report_latest.json`

### 第四阶段：Step5 / Step6 收口

目标：`prob_final` 稳定、`final_score` 不退化
产物：稳定 Top10 输出链

### 第五阶段：训练发布安全化

目标：staging → release → rollback 体系成型
产物：正式模型更新链可控

### 第六阶段：命中评估深化

目标：从“会输出”升级到“会审判自己”
产物：分层命中率 / 分场景评估

## 14.2 阶段内执行纪律

每阶段必须遵守：

1. 先写契约
2. 再改代码
3. 再手动验收
4. 验收通过后再进下一阶段

禁止：

* 同时大改多个层级导致不可定位
* 先写代码后补契约
* 还未验收就进入下一主线

---

## 15. 当前第一优先任务（V3 开工点）

### 15.1 第一优先任务定义

当前 V3 的第一优先任务不是改模型，而是：

> **输出 FeatureHistory 字段入池映射表 V1**

该表必须覆盖：

* 字段英文名
* 来源表
* 主键
* 落地逻辑
* 缺失策略
* 闸门阈值
* 所属阶段
* 是否主特征

### 15.2 第一优先任务完成标准

完成后必须能直接指导开发：

* 哪些字段先写
* 从哪张表取
* 空了怎么办
* 验收看什么
* 哪些字段不能进

---

## 16. 版本治理

## 16.1 本文档地位

本文档作为：

> **a-top10 V3 正式需求主文件**

后续新增文档可以作为子文档，但不得与本文档冲突。

## 16.2 后续子文档建议

后续可拆分：

1. `FeatureHistory 字段入池映射表 V1.MD`
2. `Step7 质量闸门契约 V1.MD`
3. `模型发布与回滚契约 V1.MD`
4. `V3 分阶段验收清单.MD`

---

## 17. 最终结论

V3 的核心不是“再加一个模型”，也不是“继续补跑历史”，而是：

> **把 a-top10 从一个能跑的预测工程，升级成一个能持续学习、可安全升级、可严谨验收的机构级涨停预测系统。**

当前最关键的突破口已经明确：

> **先把上游字段契约盘清，再把 FeatureHistory 做厚，再让 Step5/Step7 真正开始稳定成长。**

在此之前，系统不能宣布完全成熟。
