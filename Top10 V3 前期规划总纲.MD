# 《Top10 V3 前期规划总纲.MD》

---

# 0. 文档定位

本文件用于合并并锁定 Top10 V3 前期规划的五个核心章节，作为后续开发、验收、收口、回看时的统一锚点。

本总纲包含以下五章：

1. 《FeatureHistory 质量闸门 V1》
2. 《标签定义与样本成熟规则 V1》
3. 《Top10 排序目标函数与评估体系 V1》
4. 《数据源修复优先级清单 V1》
5. 《V3 阶段执行路线图与验收关卡 V1》

本文件目标不是“写给人看着舒服”，而是作为后续开发执行时的**唯一前期规划总锚点**。

---

# 1. 使用原则

## 1.1 总原则

后续凡涉及以下事项，优先回看本文件：

- feature_history 字段放行
- 标签定义
- 样本成熟规则
- 排序目标函数
- 评估指标
- 数据源修复优先级
- V3 阶段推进与验收关卡

## 1.2 优先级原则

若后续代码、旧文档、临时讨论与本文件冲突，以本文件为准。

## 1.3 修改原则

本文件后续允许升级到 V2 / V3，但不得无记录修改。  
任何调整必须显式更新版本并说明原因。

---

# 第一章 《FeatureHistory 质量闸门 V1》

## 1. 文档目的

本章用于定义 `outputs/learning/feature_history.csv` 的质量放行规则，解决以下问题：

1. 哪些样本可以进训练。
2. 哪些样本只能审计，不能训练。
3. 哪些交易日整批必须阻断。
4. 当关键字段缺失时，系统该降级还是拦截。
5. 如何避免“脏样本进入训练后污染 Step5 / Step7 学习链”。

本章与《FeatureHistory 字段入池映射表 V1》配套使用：

- 字段入池映射表：解决“字段是什么、从哪来”
- 质量闸门：解决“字段到了以后，能不能放行”

---

## 2. 适用范围

本闸门适用于以下链路：

- `Step3 Strength Score`
- `Step4 Theme Boost`
- `Step5 ML Probability`
- `Step7 Self Learning`
- `feature_history.csv`
- 训练集构造过程
- 历史补跑样本
- 实时运行样本

---

## 3. 质量闸门的总体原则

### 3.1 原则一：宁缺毋脏

训练集允许样本量暂时偏少，  
但不允许脏字段、伪字段、未成熟标签进入训练。

### 3.2 原则二：先保样本口径，再谈模型复杂度

若样本质量不稳，则：

- LR 不会可靠
- LGBM 不会可靠
- 融合概率不会可靠
- 排序结果不会稳定

### 3.3 原则三：闸门分三级

质量闸门分为：

1. **字段级闸门**
2. **样本级闸门**
3. **交易日批级闸门**

### 3.4 原则四：任何拦截必须留下痕迹

每次拒绝训练、拒绝入池、拒绝整批放行，都必须留下：

- `learnable_flag`
- `reject_reason`
- `gate_level`
- `gate_version`

---

## 4. 闸门分级定义

### 4.1 字段级闸门

判断单个字段是否合格。

### 4.2 样本级闸门

判断某一行样本是否可训练。

### 4.3 批级闸门

判断某一 `trade_date` 的整批样本是否可训练。

---

## 5. 字段级闸门

### 5.1 一级硬字段（缺失即不可训练）

以下字段任一缺失，则该样本 `learnable_flag = 0`：

- `trade_date`
- `ts_code`
- `close`
- `StrengthScore`
- `ThemeBoost`
- `Probability`
- `_prob_src`
- `is_sample_mature`
- `y_limit_hit`

### 处理规则

- 单字段缺失：该样本拒绝训练
- 多字段缺失：该样本拒绝训练，并累计为批级风险信号

---

### 5.2 二级核心字段（大面积缺失将阻断整批）

以下字段不是单条必杀，但高缺失会导致整批失真：

- `turnover_rate`
- `open_times`
- `seal_amount`
- `limit_type`
- `up_limit`
- `is_limit_up_pool`
- `theme_heat`
- `lhb_flag`

### 处理规则

- 单条缺失：允许保留样本，但记录风险
- 整批高缺失：触发批级阻断

---

### 5.3 三级增强字段（允许暂缺，不阻断主链）

- `first_limit_time`
- `last_limit_time`
- `limit_open_count`
- `lhb_net_buy`
- `north_money_flag`
- `board_rank`
- `volume_ratio`

### 处理规则

- 单条缺失：允许
- 整批缺失：只告警，不阻断

---

## 6. 样本级闸门

### 6.1 样本放行最低条件

一个样本只有同时满足以下条件，才能进入训练：

1. `trade_date` 非空
2. `ts_code` 非空
3. `is_sample_mature = 1`
4. `StrengthScore` 非空
5. `ThemeBoost` 非空
6. `Probability` 非空
7. `_prob_src` 非空
8. `y_limit_hit` 非空

不满足任一条件：

- `learnable_flag = 0`
- 写入 `reject_reason`

---

### 6.2 微观盘口簇条件

以下字段共同构成“微观盘口簇”：

- `turnover_rate`
- `open_times`
- `seal_amount`

若三者全部缺失，则样本不得训练。

### 原因

如果这三项全空，样本退化为：

- 静态价格壳
- 弱行为特征
- 弱盘口结构

这种样本即使入训，也会稀释模型真实判别力。

### 处理规则

- 三者全空：拒绝训练
- 任意一项非空：允许进入候审
- 至少两项非空：视为正常质量样本

---

### 6.3 样本质量等级

每条样本进入训练前，定义质量等级：

#### A级样本
满足：

- 一级硬字段全部齐全
- 微观盘口簇至少 2 项非空
- 题材热度层核心字段基本齐全

#### B级样本
满足：

- 一级硬字段全部齐全
- 微观盘口簇至少 1 项非空

#### C级样本
满足：

- 一级硬字段齐全
- 微观盘口簇仅 0 或 1 项弱可用
- 但尚未触发拒绝条件

#### D级样本
不满足放行条件，拒绝训练

---

### 6.4 样本级 reject_reason 规范

建议统一使用以下值：

- `missing_trade_date`
- `missing_ts_code`
- `missing_close`
- `missing_strengthscore`
- `missing_themeboost`
- `missing_probability`
- `missing_prob_src`
- `sample_not_mature`
- `missing_label`
- `missing_microstructure_cluster`
- `corrupt_row`
- `duplicate_sample`
- `invalid_trade_date`

---

## 7. 批级闸门

### 7.1 交易日整批阻断条件

某个 `trade_date` 满足以下任一条件，则当天样本不得进入正式训练集：

1. `StrengthScore` 缺失率 > 30%
2. `ThemeBoost` 缺失率 > 30%
3. `Probability` 缺失率 > 20%
4. `turnover_rate` 缺失率 > 40%
5. `open_times` 缺失率 > 70%
6. `seal_amount` 缺失率 > 70%
7. 成熟样本数 < 最低训练门槛
8. 标签成熟规则异常
9. 真值来源异常或不完整

---

### 7.2 样本量门槛

定义三档：

#### Level 1：最低可学习门槛
- `mature_learnable_samples >= 20`

#### Level 2：基本可信门槛
- `mature_learnable_samples >= 50`

#### Level 3：稳健训练门槛
- `mature_learnable_samples >= 100`

### 放行规则

- `< 20`：不得训练，只审计
- `20 ~ 49`：允许弱训练，低权重
- `50 ~ 99`：允许正常训练
- `>= 100`：允许稳健训练

---

### 7.3 批级质量评分

建议为每个交易日计算 `batch_quality_score`，范围 0~100：

#### 分数组成建议

- 一级硬字段完整度：30分
- 微观盘口簇完整度：25分
- 题材层完整度：15分
- 标签完整度：20分
- 样本量等级：10分

### 执行阈值建议

- `>= 80`：正常训练
- `60 ~ 79`：可训练但低权重
- `< 60`：不得进入正式训练

---

## 8. 历史补跑与实时样本统一规则

### 8.1 统一原则

历史补跑与实时样本必须执行同一套：

- 字段映射
- 标签成熟
- 质量闸门
- 放行逻辑

禁止出现：

- 历史补跑口径一套
- 实时运行口径另一套

---

### 8.2 历史样本例外处理

若历史样本存在结构性数据缺口：

- 可以进入审计层
- 可以进入“待补数仓”队列
- 不得直接进入正式训练集

---

## 9. 输出字段建议

建议在 `feature_history.csv` 或配套审计表中增加：

- `gate_version`
- `gate_level`
- `sample_quality_grade`
- `batch_quality_score`
- `learnable_flag`
- `reject_reason`

---

## 10. 执行优先级

### 第一优先
把以下闸门写进 Step7：

- 一级硬字段检查
- 样本成熟检查
- 微观盘口簇检查
- 批级阻断检查

### 第二优先
补：

- 样本质量等级
- 批级质量评分
- 历史/实时统一日志

---

## 11. V1 结论

本章作用：

1. 把“能不能训练”从主观判断改成规则判断
2. 把“样本污染”挡在训练集外
3. 让 Step7 不再只是落库，而是真正成为质量控制中枢

---

# 第二章 《标签定义与样本成熟规则 V1》

## 1. 文档目的

本章用于锁定 Top10 系统的：

1. 主标签定义
2. 辅标签定义
3. 样本成熟条件
4. 历史补跑与实时运行的统一口径

目标是避免 Step7 学习链在标签层出现漂移。

---

## 2. 系统最高目标

Top10 的最高目标不是泛泛预测上涨，而是：

> **找出 T+1 最大概率涨停的股票，并把它们尽可能排在前列。**

所以主标签必须服务这个目标。

---

## 3. 标签体系总览

标签体系分为三层：

1. **主标签**
2. **辅标签**
3. **扩展行为标签**

---

## 4. 主标签定义

### 4.1 主标签名称

`y_limit_hit`

### 4.2 主标签含义

表示：

> **T 日预测的股票，在 T+1 是否达到“涨停命中”定义。**

---

### 4.3 V1 主标签的正式定义

V1 阶段，主标签统一定义为：

> **若股票在 T+1 交易日收盘处于涨停状态，则 `y_limit_hit = 1`；否则为 0。**

即：

- 收盘涨停：1
- 盘中触板但收盘未封住：0
- 盘中大涨但未涨停：0

### 选择这个定义的原因

1. 口径硬
2. 真值易统一
3. 不容易因盘口细节歧义漂移
4. 与“高质量涨停命中”更接近

---

### 4.4 暂不采用的标签定义

以下定义暂不作为主标签：

- 盘中触板即算命中
- 次日最大涨幅超过某阈值
- 次日涨幅排名进入前N
- 次日炸板后回封算半命中

这些可作为后续辅标签或 V2 扩展标签。

---

## 5. 辅标签定义

### 5.1 次日收益率

字段名：

`y_next_ret`

定义：

> T+1 收盘相对于 T 日收盘的收益率

作用：

- 评估非涨停但高收益样本
- 为后续排序优化提供辅助监督

---

### 5.2 次日高点收益率

字段名建议：

`y_next_high_ret`

定义：

> T+1 最高价相对于 T 日收盘的收益率

作用：

- 衡量“曾经很强但未封死”的票
- 为后续行为标签扩展做准备

---

### 5.3 次日触板标签

字段名建议：

`y_touch_limit`

定义：

> T+1 是否曾触及涨停价

作用：

- 区分“完全没到涨停区”和“到过但没封住”

V1 阶段可不作为主标签，但建议预留。

---

## 6. 样本成熟规则

### 6.1 样本成熟的基本定义

对于 T 日样本：

> 当系统已经能稳定取得 T+1 的真值数据时，该样本才算成熟。

字段：

`is_sample_mature`

- 成熟：1
- 未成熟：0

---

### 6.2 成熟判定条件

T 日样本成熟，至少满足：

1. 已进入 T+1 之后
2. T+1 行情主表可得
3. T+1 涨停真值表可得
4. 标签计算未报错
5. 标签口径符合 V1 契约

---

### 6.3 未成熟样本规则

若样本未成熟：

- `is_sample_mature = 0`
- `y_limit_hit = 空`
- `y_next_ret = 空`
- 不得进入训练集
- 允许保留在 feature_history 中等待后续补成熟

---

### 6.4 真值晚到规则

若 T+1 真值当天缺失或不完整：

- 该批样本延迟成熟
- 不得强行打 0
- 不得强行入训

---

## 7. 历史补跑成熟规则

历史补跑样本只有在以下条件成立时才可成熟：

1. T+1 交易日行情可得
2. T+1 标签真值可得
3. 数据源字段完整度达到最低质量线
4. 没有发生交易日日历错位

---

## 8. 标签优先级

### 8.1 主标签优先级最高

Step5 / Step7 训练主目标优先围绕：

- `y_limit_hit`

### 8.2 辅标签作为增强监督

后续可逐步引入：

- `y_next_ret`
- `y_next_high_ret`
- `y_touch_limit`

但不得在 V1 阶段破坏主标签口径统一。

---

## 9. 标签风险控制

### 9.1 禁止未来信息污染

不得使用任何 T+1 之后的数据来定义 T+1 主标签。

### 9.2 禁止标签口径漂移

不同日期、不同批次、不同补跑方式，主标签口径必须一致。

### 9.3 禁止未成熟样本混训

这是硬规则。

---

## 10. 建议新增字段

建议在 feature_history 或真值表中保留：

- `label_version`
- `label_source`
- `label_generated_at`
- `is_sample_mature`
- `mature_reason`
- `label_delay_flag`

---

## 11. V1 结论

本章作用：

1. 锁死主标签，不再摇摆
2. 锁死成熟规则，不让 Step7 漂移
3. 为后续多标签扩展留好接口

---

# 第三章 《Top10 排序目标函数与评估体系 V1》

## 1. 文档目的

本章定义：

1. Top10 系统真正优化的目标
2. 排序目标函数口径
3. 评估指标体系
4. 验收标准

核心问题不是“模型有没有概率输出”，而是：

> **最终 Top10 排序，是否把最该排前的票排在前面。**

---

## 2. 优化目标

### 2.1 一级目标

> 最大化 Top10 名单中 T+1 收盘涨停命中数。

---

### 2.2 二级目标

> 在命中数相近时，让真正命中票尽量排在更前的位置。

---

### 2.3 三级目标

> 让系统表现具备滚动稳定性，而不是靠单日偶然命中。

---

## 3. 排序对象

排序对象不是全市场所有股票的绝对价格预测，  
而是经过 Step2 候选池筛选后的样本池。

最终输出：

- Top10 排名
- Top10 之外的全量排序表

---

## 4. V1 排序目标函数

建议定义最终综合排序分：

`RankScore`

V1 阶段建议结构：

`RankScore = w1 * Probability + w2 * StrengthScore + w3 * ThemeBoost + w4 * QualityAdj`

其中：

- `Probability`：主导项
- `StrengthScore`：盘口强度补充
- `ThemeBoost`：题材热度补充
- `QualityAdj`：质量修正项

---

## 5. 排序原则

### 5.1 Probability 是主轴

没有可靠概率，排序会失去主心骨。

### 5.2 StrengthScore 用于拉开“同概率票”的差距

特别是在高概率区，盘口强弱差异必须参与排序。

### 5.3 ThemeBoost 用于修正题材共振效应

同等盘口下，题材热度更强的票，应获得适度排序优势。

### 5.4 样本质量差的票应被轻微打折

若字段质量、来源质量、微观盘口完整度偏弱，应施加质量折减。

---

## 6. 评估体系

评估分四类：

### 6.1 命中能力指标
- `Hit@10`：Top10 中命中数量
- `Hit@5`
- `Hit@3`
- `Precision@10`
- `Precision@5`
- `Precision@3`

### 6.2 排序质量指标
- 命中样本平均排名
- Top10 中命中票是否集中在前5
- NDCG@10
- MRR（可选）

### 6.3 稳定性指标
- 5日滚动 Hit@10
- 10日滚动 Hit@10
- 20日滚动 Hit@10
- 标准差 / 波动系数

### 6.4 数据质量指标
- 一级字段缺失率
- 可训练样本占比
- 批级阻断率
- 标签延迟率

---

## 7. V1 验收标准

### 7.1 单日验收不作为最终标准

单日偶然命中不代表系统变强。

### 7.2 滚动窗口验收为主

建议以：

- 5日
- 10日
- 20日

三个窗口联合判断。

---

### 7.3 升级验收建议

若某次升级后，满足以下至少两项，则可判定为有效升级：

1. `Hit@10` 滚动均值提升
2. `Hit@5` 滚动均值提升
3. NDCG@10 提升
4. 命中票平均排名前移
5. 稳定性未显著恶化

---

## 8. 排序验收的反作弊原则

以下情况不得判定为升级成功：

- 只看 1 天结果
- 只看 1 次历史补跑
- 样本口径变化后直接横比
- 数据质量下降但指标偶然上涨
- 通过压缩候选池获得虚高命中率

---

## 9. 报表建议

建议系统长期产出以下评估表：

- `rank_eval_daily.csv`
- `rank_eval_rolling.csv`
- `hit_distribution.csv`
- `feature_quality_daily.csv`

---

## 10. V1 结论

本章意义：

1. 明确系统不是做“有概率就行”
2. 明确最终要优化的是 Top10 排序质量
3. 以后升级都能按同一套指标验收

---

# 第四章 《数据源修复优先级清单 V1》

## 1. 文档目的

本章用于锁定 `a-share-top3-data` 及相关上游数据源的修复优先级，避免开发注意力被模型细节吸走，而忽略真正决定样本上限的数据底座。

---

## 2. 总原则

### 2.1 数据优先于调参

当前 Top10 的主要瓶颈，不在 Step5 公式花样，  
而在微观盘口字段缺失与不稳定。

### 2.2 先修“决定样本判别力”的字段

优先修：

- `open_times`
- `seal_amount`
- `limit_type`
- `up_limit`
- `down_limit`

### 2.3 先修稳定来源，再修增强来源

---

## 3. 第一优先级（必须修）

### 3.1 `limit_break_d.csv`

#### 当前问题
- 为空表
- 导致 `open_times` / `limit_open_count` 缺失

#### 影响
- 微观盘口簇严重残缺
- Step3 / Step7 判别力下降

#### 修复目标
- 稳定生成 `open_times`
- 若可能，生成更细炸板行为字段

---

### 3.2 `limit_list_d.csv`

#### 当前问题
- 字段残缺
- `limit_type` / `up_limit` / `down_limit` / `seal_amount` 不完整

#### 影响
- 涨停行为口径不稳
- 标签和强度逻辑受损

#### 修复目标
- 补齐核心字段
- 统一空值口径
- 确保历史与实时一致

---

### 3.3 `stk_limit.csv`

#### 当前角色
- 作为 `up_limit/down_limit` 回退源

#### 修复目标
- 确保与主表代码、日期对齐
- 可稳定用于回退

---

## 4. 第二优先级（重要增强）

### 4.1 首次封板时间 / 最后封板时间源表

目标：

- 支撑时序盘口特征
- 提升强弱区分力

### 4.2 开板细分行为

目标：

- 区分“一次炸板”和“多次反复开板”
- 提升脆弱板识别力

---

## 5. 第三优先级（结构增强）

- 龙虎榜细项
- 北向关注映射
- 更细板块热度
- 题材热度层级结构

---

## 6. 修复顺序

### Phase A
- `limit_break_d.csv`
- `limit_list_d.csv`
- `stk_limit.csv`

### Phase B
- 时序盘口源
- 首封/末封源
- 开板细分源

### Phase C
- 龙虎榜增强
- 北向增强
- 板块热度增强

---

## 7. 每个源表必须补的内容

每张表修复时必须明确：

1. 表头契约
2. 主键
3. 日期口径
4. 缺失口径
5. 历史回放一致性
6. 下游消费文件

---

## 8. 修复验收标准

### 合格条件
- 字段可长期稳定产出
- 历史样本与实时样本口径一致
- 缺失率显著下降
- 下游 feature_history 可真实入池

---

## 9. V1 结论

本章作用：

1. 明确现在最该修的是数据，不是花式调参
2. 让数仓修复顺序不漂移
3. 为后续 Step3/Step7 提供可靠底座

---

# 第五章 《V3 阶段执行路线图与验收关卡 V1》

## 1. 文档目的

本章用于把 Top10 V3 从“讨论规划”转为“阶段执行”。

核心目标：

1. 明确每阶段做什么
2. 明确阶段验收关卡
3. 明确不达标不得进入下一阶段

---

## 2. V3 总体阶段划分

- Phase A：数据底座修复
- Phase B：样本链路收口
- Phase C：排序目标优化
- Phase D：自学习闭环强化
- Phase E：稳定运行与迭代升级

---

## 3. Phase A：数据底座修复

### 目标
修复 `a-share-top3-data` 中对 Top10 判别力最关键的源表问题。

### 重点任务
- 修 `limit_break_d.csv`
- 修 `limit_list_d.csv`
- 校验 `stk_limit.csv`
- 确保 `open_times / seal_amount / limit_type / up_limit / down_limit` 可稳定获取

### 验收关卡
1. 一级源表契约锁死
2. 历史日期可回放
3. 关键字段缺失率降到可控范围
4. feature_history 能真实吃到关键字段

---

## 4. Phase B：样本链路收口

### 目标
把 `feature_history.csv` 从流水表升级为正式训练样本契约表。

### 重点任务
- 写入 `StrengthScore`
- 写入 `ThemeBoost`
- 写入 `Probability`
- 写入 `_prob_src`
- 写入 `is_sample_mature`
- 写入 `learnable_flag`
- 写入 `reject_reason`

### 验收关卡
1. 字段映射按契约落地
2. 质量闸门真实执行
3. 标签成熟规则真实执行
4. 历史样本与实时样本口径统一

---

## 5. Phase C：排序目标优化

### 目标
从“概率输出系统”升级为“Top10 排序系统”。

### 重点任务
- 建立 `RankScore`
- 统一排序主逻辑
- 引入质量修正项
- 优化前10排序而非仅分类

### 验收关卡
1. 排序函数固定
2. Top10 / Top5 / Top3 指标可量化
3. 命中票平均排名前移
4. 不因口径变化造成虚高改进

---

## 6. Phase D：自学习闭环强化

### 目标
让 Step7 真正成为训练、评估、反馈的闭环中枢。

### 重点任务
- 训练集构造规范化
- 批级质量评分
- 多日滚动评估
- 训练版本记录
- 推理模型版本记录

### 验收关卡
1. 训练—推理—评估链打通
2. 历史补跑可形成稳定样本积累
3. 模型更新可追踪
4. 滚动指标可观测

---

## 7. Phase E：稳定运行与迭代升级

### 目标
进入持续迭代状态，而不是继续大面积返工。

### 重点任务
- 建立固定验收节奏
- 建立升级回归测试
- 小步增强新增字段
- 逐步引入更高阶行为标签

### 验收关卡
1. 连续多日稳定运行
2. 无重大字段漂移
3. 无标签成熟漂移
4. 滚动指标稳定

---

## 8. 阶段切换规则

### 可切换到下一阶段的条件
- 当前阶段核心任务完成
- 当前阶段验收关卡通过
- 未解决问题已记录并不阻断主线

### 不得切换的情况
- 数据底座未稳
- feature_history 仍只是流水表
- 标签成熟口径未锁
- 排序目标未定义
- 评估体系缺失

---

## 9. 每阶段必须沉淀的产物

- 契约文档
- 验收文档
- 关键日志
- 评估表
- 版本记录

---

## 10. V1 结论

本章作用：

1. 把 V3 变成可推进工程
2. 防止开发顺序错乱
3. 防止“主线没跑完就跑偏”

---

# 结尾：总纲结论

《Top10 V3 前期规划总纲.MD》形成后，Top10 V3 前期规划层面已完成以下五个关键闭环：

1. **样本怎么放行**：由质量闸门锁定
2. **标签怎么定义**：由标签与成熟规则锁定
3. **排序优化什么**：由目标函数与评估体系锁定
4. **数据先修什么**：由数据源优先级清单锁定
5. **阶段怎么推进**：由阶段路线图与验收关卡锁定

这意味着后续开发不再是“边讨论边猜”，而是进入：

> **按总纲推进、按章节执行、按关卡验收、按指标收口**

---

# 后续直接动作

本总纲落库后，下一步最合理的是继续输出：

## 动作 A
《Top10 V3 前期规划总纲》对应的**代码改造总清单**

## 动作 B
把总清单拆到具体文件：

- `a_top10/steps/step3_strength_score.py`
- `a_top10/steps/step4_theme_boost.py`
- `a_top10/steps/step5_ml_probability.py`
- `a_top10/steps/step6_final_topn.py`
- `a_top10/steps/step7_self_learning.py`
- `a_top10/io/writers.py`
- `a_top10/main.py`
- 必要的 `schema / config / outputs` 相关文件

## 动作 C
按 Phase A → Phase B → Phase C 的顺序正式进入 V3 开发与验收

---

**文档名称**：`《Top10 V3 前期规划总纲.MD》`  
**版本**：V1  
**定位**：Top10 V3 前期统一锚点  
**状态**：可立即落库并作为后续开发总依据
# 《Top10 V3 前期规划总纲——代码改造总清单 V1》

---

# 0. 文档定位

本文件是《Top10 V3 前期规划总纲.MD》的直接执行层文档。  
作用不是补充讨论，而是把总纲中的规则，拆成**可以直接落到文件修改、函数修改、字段落库、验收检查**的代码改造清单。

本清单服务目标：

1. 把 V3 总纲映射到具体文件。
2. 明确每个文件在 V3 中承担什么职责。
3. 明确每个文件要新增什么、删除什么、重构什么。
4. 明确哪些改动属于 Phase A / B / C。
5. 明确开发顺序，避免多文件同时乱改导致主线漂移。

---

# 1. 执行总原则

## 1.1 先主链，后增强

本轮不是做“全面美化重构”，而是先把以下主链锁死：

- 数仓关键字段可得
- feature_history 成为正式训练样本表
- Step7 真实执行质量闸门与样本成熟规则
- Step6 从简单排序升级到明确 RankScore 排序

---

## 1.2 先写入，再优化

优先级顺序：

1. 先把字段写对
2. 先把标签写对
3. 先把闸门拦住
4. 先把排序跑通
5. 再谈模型增强与权重微调

---

## 1.3 不允许“文档在，代码没接”

总纲中的以下内容，必须有真实代码承接：

- `learnable_flag`
- `reject_reason`
- `is_sample_mature`
- `y_limit_hit`
- `RankScore`
- `batch_quality_score`
- 样本级 / 批级闸门

---

## 1.4 修改方式原则

后续代码修改时统一遵守：

- 若我记得原文件结构，则直接输出完整替换文件
- 若结构不清晰，则先从仓库文件现状出发再给完整文件
- 不做零碎片段式补丁为主线交付方式
- 每改一个主文件，都要说明：
  - 这个文件为什么改
  - 改了什么
  - 如何验收

---

# 2. Phase A / B / C 总体文件映射

| 阶段 | 核心目标 | 主要文件 |
|---|---|---|
| Phase A | 数据底座修复对接 | `a_top10/main.py`、`step2_candidate_pool.py`、`step3_strength_score.py`、必要的数据读取/工具模块 |
| Phase B | 样本链路收口 | `step3_strength_score.py`、`step4_theme_boost.py`、`step5_ml_probability.py`、`step7_self_learning.py`、`io/writers.py` |
| Phase C | 排序目标优化 | `step5_ml_probability.py`、`step6_final_topn.py`、`io/writers.py`、`main.py` |

---

# 3. 文件级改造总表

---

## 3.1 `a_top10/main.py`

### 当前职责
主程序编排入口，负责串联 Step2 ~ Step7。

### V3 目标职责
升级为 V3 主链调度器，显式控制：

- 候选池生成
- StrengthScore 计算
- ThemeBoost 计算
- Probability 计算
- RankScore 排序
- feature_history 写入
- 样本成熟与质量闸门
- Step7 学习闭环

### 必改内容

#### A. 显式定义 V3 主链执行顺序
建议主链顺序固定为：

1. Step2 Candidate Pool
2. Step3 Strength Score
3. Step4 Theme Boost
4. Step5 ML Probability
5. Step6 Final TopN / RankScore
6. Writers 落库
7. Step7 Self Learning / Gate / Label

#### B. 增加 V3 运行上下文字段
建议统一在主链上下文中传递：

- `trade_date`
- `run_id`
- `version_tag`
- `gate_version`
- `label_version`
- `rank_version`

#### C. 增加失败拦截点
主程序必须能在以下场景中止后续训练写入：

- 核心中间表为空
- 一级关键字段大面积缺失
- 标签未成熟但被误送训练
- Step6 无法生成最终排名结果

#### D. 增加阶段日志
至少打印：

- 候选池数量
- StrengthScore 可用率
- ThemeBoost 可用率
- Probability 可用率
- feature_history 入池数
- mature 样本数
- learnable 样本数
- 被拒绝样本数

### 验收点

1. 主链执行顺序固定且日志可见
2. Step3~Step7 中间结果都被主程序真实消费
3. 若闸门失败，主程序能阻断训练写入
4. 运行结果能区分“预测产物”和“训练产物”

---

## 3.2 `a_top10/steps/step2_candidate_pool.py`

### 当前职责
生成候选池。

### V3 目标职责
继续负责候选池，但要为后续排序和学习链提供**稳定主键样本池**。

### 必改内容

#### A. 候选池输出字段标准化
至少保证输出：

- `trade_date`
- `ts_code`
- `name`
- `close`
- `pct_chg`
- `amount`
- `vol`
- `turnover_rate`
- `circ_mv`
- `total_mv`
- `is_st`
- `list_days`
- `board_type`

#### B. 候选池主键唯一化
要求：

- `trade_date + ts_code` 唯一
- 无重复行
- 无空代码行

#### C. 候选池异常过滤显式化
建议明确：

- ST 剔除逻辑
- 新股过滤逻辑
- 异常停牌 / 缺价样本过滤逻辑

### 验收点

1. 候选池行唯一
2. 主键字段稳定
3. 进入 Step3 的基础字段完整率可统计

---

## 3.3 `a_top10/steps/step3_strength_score.py`

### 当前职责
计算强度分。

### V3 目标职责
成为“盘口强弱特征聚合器 + StrengthScore 输出器”。

### 必改内容

#### A. 明确输入字段来源
优先接入：

- `turnover_rate`
- `limit_type`
- `up_limit`
- `is_limit_up_pool`
- `open_times`
- `seal_amount`
- 可得时再接：
  - `first_limit_time`
  - `last_limit_time`
  - `limit_open_count`

#### B. 区分原始特征与最终得分
必须区分：

- `limit_strength_raw`
- `StrengthScore`

不要把原始分和最终分混为一个字段。

#### C. 缺失策略写死
必须遵守《FeatureHistory 字段入池映射表 V1》：

- `open_times` 缺失：置空，不得伪造0
- `seal_amount` 缺失：置空，不得伪造0
- `turnover_rate` 缺失：置空，不得伪造0

#### D. 输出质量标记
建议额外输出：

- `strength_feature_count`
- `strength_quality_flag`

用于 Step7 判断样本质量。

### 验收点

1. `StrengthScore` 真正写入下游
2. 缺失字段未被伪造为0
3. `StrengthScore` 与原始强度字段可区分
4. 输出可进入 feature_history

---

## 3.4 `a_top10/steps/step4_theme_boost.py`

### 当前职责
计算题材加权分。

### V3 目标职责
成为“题材热度特征聚合器 + ThemeBoost 输出器”。

### 必改内容

#### A. 明确输入字段
至少支持：

- `theme_main`
- `theme_count`
- `theme_heat`
- `board_rank`
- `lhb_flag`
- `lhb_net_buy`
- `north_money_flag`

#### B. ThemeBoost 输出标准化
必须显式输出：

- `ThemeBoost`
- `theme_feature_count`
- `theme_quality_flag`

#### C. 题材缺失策略显式化
- `theme_heat` 缺失：置空
- `lhb_flag` 缺失：可置0
- `lhb_net_buy` 缺失：置空

### 验收点

1. `ThemeBoost` 稳定输出
2. 题材层核心字段可统计缺失率
3. 输出可直接写入 feature_history

---

## 3.5 `a_top10/steps/step5_ml_probability.py`

### 当前职责
输出涨停概率。

### V3 目标职责
成为“主概率引擎 + 概率来源记录器 + 排序输入主轴”。

### 必改内容

#### A. 保留多来源概率
建议输出：

- `prob_lr`
- `prob_lgbm`
- `prob_rule`
- `Probability`
- `_prob_src`

#### B. 增加概率融合规则显式化
必须能说明最终 Probability 来自：

- `lr`
- `lgbm`
- `blend`
- `fallback_rule`

#### C. 增加样本质量折扣接口
为后续 RankScore 预留：

- `quality_adj`
- 或可用于 Step6 再计算

#### D. 概率缺失阻断
若 `Probability` 无法得到，则：

- 该样本不得入正式训练
- TopN 排序不得把该样本当正常样本参与

### 验收点

1. `Probability` 与 `_prob_src` 同时存在
2. fallback 与 blend 路径可区分
3. 下游 Step6 能直接消费 Probability

---

## 3.6 `a_top10/steps/step6_final_topn.py`

### 当前职责
生成最终 TopN。

### V3 目标职责
从“简单选前10”升级为“显式 RankScore 排序器”。

### 必改内容

#### A. 引入 `RankScore`
建议结构先固定为：

`RankScore = w1 * Probability + w2 * StrengthScore + w3 * ThemeBoost + w4 * QualityAdj`

V1 阶段先允许权重写在配置或文件内，先锁口径。

#### B. 输出排序解释字段
至少输出：

- `RankScore`
- `rank_final`
- `in_topn`

#### C. 全量排序与 Top10 输出分离
要求：

- 输出全量排序表
- 输出 Top10 子集
- 两者口径一致，不得各算各的

#### D. 同分处理策略
若 `RankScore` 接近或相同，建议按以下顺序破同分：

1. `Probability`
2. `StrengthScore`
3. `ThemeBoost`
4. `ts_code` 或稳定顺序键

### 验收点

1. RankScore 已真实参与排序
2. Top10 与全量排序一致
3. `rank_final`、`in_topn` 能落到 feature_history

---

## 3.7 `a_top10/steps/step7_self_learning.py`

### 当前职责
自学习与训练闭环。

### V3 目标职责
升级为“样本成熟判定器 + 质量闸门执行器 + 标签写入器 + 训练放行器”。

### 必改内容

#### A. 写入成熟度逻辑
必须真实生成：

- `is_sample_mature`
- `mature_reason`
- `label_delay_flag`（可选）

#### B. 写入标签
至少支持：

- `y_limit_hit`
- `y_next_ret`

建议预留：

- `y_next_high_ret`
- `y_touch_limit`

#### C. 执行样本级闸门
根据总纲，至少检查：

- 一级硬字段是否齐全
- `Probability` 是否存在
- `StrengthScore` 是否存在
- `ThemeBoost` 是否存在
- `y_limit_hit` 是否成熟
- `turnover_rate/open_times/seal_amount` 是否全空

#### D. 执行批级闸门
至少计算：

- `StrengthScore` 缺失率
- `ThemeBoost` 缺失率
- `Probability` 缺失率
- `turnover_rate` 缺失率
- `open_times` 缺失率
- `seal_amount` 缺失率
- `mature_learnable_samples`

#### E. 生成闸门字段
必须写：

- `learnable_flag`
- `reject_reason`
- `sample_quality_grade`
- `batch_quality_score`
- `gate_version`

### 验收点

1. Step7 不再只是简单落库
2. `learnable_flag` 和 `reject_reason` 真实存在
3. 样本成熟与训练放行分开
4. 批级阻断能真正生效

---

## 3.8 `a_top10/io/writers.py`

### 当前职责
写出对外产物与学习产物。

### V3 目标职责
成为“V3 契约落库层”。

### 必改内容

#### A. feature_history 字段落库统一
必须支持写入以下核心字段：

- `trade_date`
- `ts_code`
- `name`
- `close`
- `turnover_rate`
- `StrengthScore`
- `ThemeBoost`
- `Probability`
- `_prob_src`
- `RankScore`
- `rank_final`
- `in_topn`
- `is_sample_mature`
- `y_limit_hit`
- `y_next_ret`
- `learnable_flag`
- `reject_reason`
- `sample_quality_grade`
- `batch_quality_score`
- `created_at`

#### B. 写入顺序固定化
要求：

- 列顺序稳定
- 同字段名长期不漂移
- 历史补跑与实时运行同口径

#### C. 支持 append-only 与去重策略
对 `feature_history.csv` 建议：

- 以 `trade_date + ts_code` 为去重主键
- 避免重复写入脏行

### 验收点

1. 核心 V3 字段可真实落库
2. CSV 列结构稳定
3. 历史补跑不产生重复脏样本

---

## 3.9 配置文件（如 `configs/default.yml`）

### V3 目标职责
承载可配置但必须锁口径的参数。

### 必改内容

建议增加：

- `gate_version`
- `label_version`
- `rank_version`
- `min_samples_level1`
- `min_samples_level2`
- `min_samples_level3`
- `rank_weights`
- `batch_quality_thresholds`

### 验收点

1. 关键阈值不再散落在代码里
2. RankScore 权重有统一配置来源

---

# 4. 按阶段拆分的改造顺序

---

## 4.1 Phase A：先改哪些文件

第一批只动：

1. `step2_candidate_pool.py`
2. `step3_strength_score.py`
3. `main.py`
4. 必要配置文件

### 目标
先确保关键字段能被主链真实吃到。

---

## 4.2 Phase B：第二批改哪些文件

第二批重点改：

1. `step4_theme_boost.py`
2. `step5_ml_probability.py`
3. `step7_self_learning.py`
4. `io/writers.py`

### 目标
先让 feature_history 真正成为训练样本契约表。

---

## 4.3 Phase C：第三批改哪些文件

第三批重点改：

1. `step6_final_topn.py`
2. `step5_ml_probability.py`（配合调整）
3. `io/writers.py`
4. `main.py`

### 目标
把系统从“概率输出”拉到“Top10 排序优化”。

---

# 5. 必须新增的字段总清单

建议 V3 至少落库这些新增字段：

- `sample_id`
- `limit_strength_raw`
- `strength_feature_count`
- `strength_quality_flag`
- `theme_feature_count`
- `theme_quality_flag`
- `prob_lr`
- `prob_lgbm`
- `prob_rule`
- `_prob_src`
- `RankScore`
- `rank_final`
- `in_topn`
- `is_sample_mature`
- `mature_reason`
- `label_delay_flag`
- `y_limit_hit`
- `y_next_ret`
- `learnable_flag`
- `reject_reason`
- `sample_quality_grade`
- `batch_quality_score`
- `gate_version`
- `label_version`
- `rank_version`

---

# 6. 本清单对应的第一轮实际开发顺序

建议后续真正开工时，按下面顺序执行：

### 第一步
改 `step3_strength_score.py`

### 第二步
改 `step4_theme_boost.py`

### 第三步
改 `step5_ml_probability.py`

### 第四步
改 `step7_self_learning.py`

### 第五步
改 `io/writers.py`

### 第六步
改 `step6_final_topn.py`

### 第七步
最后收口 `main.py` 与配置文件

---

# 7. 第一轮开发后的正式验收清单

第一轮改造后，必须验证：

1. `feature_history.csv` 是否真实新增 V3 字段
2. `learnable_flag` 是否真实计算
3. `reject_reason` 是否真实写入
4. `is_sample_mature` 是否真实写入
5. `y_limit_hit` 是否真实写入
6. `Probability` 和 `_prob_src` 是否真实写入
7. `RankScore` 是否真实参与排序
8. Top10 与全量排序是否同口径
9. 历史补跑与实时样本是否同口径
10. Step7 是否能真正阻断脏样本训练

---

# 8. V1 结论

《Top10 V3 前期规划总纲——代码改造总清单 V1》的作用，是把前期五章规划，真正压到代码文件层。

这意味着从这一刻开始：

- 不是再讨论“该不该做”
- 而是已经明确“先改哪个文件、改什么、怎么验收”

后续开发必须按本清单推进，不得跳着改，不得一边改一边重新定义总纲口径。

---

**文档名称**：`《Top10 V3 前期规划总纲——代码改造总清单 V1》`  
**版本**：V1  
**定位**：V3 总纲的直接执行层  
**状态**：可立即作为后续开发排期与逐文件改造依据
