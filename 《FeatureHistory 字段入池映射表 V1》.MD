# 《FeatureHistory 字段入池映射表 V1》

## 1. 文档目的

本表用于锁定 `outputs/learning/feature_history.csv` 的**字段入池规则**，避免后续开发出现以下问题：

1. 字段名漂移，今天叫这个，明天叫那个。
2. 来源表混乱，同一字段多人理解不一致。
3. 缺失值处理随意，导致训练样本口径不稳定。
4. 上游数据源缺字段时，下游仍硬算，污染学习链。
5. Step3 / Step4 / Step5 / Step7 各自为战，无法形成统一契约。

本文件是 **FeatureHistory 的字段契约锚点**。  
后续开发、补字段、加闸门、改训练逻辑，均以本表为准。

---

## 2. 适用范围

本映射表服务于以下主链：

- `Step2 Candidate Pool`
- `Step3 Strength Score`
- `Step4 Theme Boost`
- `Step5 ML Probability`
- `Step6 Final TopN`
- `Step7 Self Learning`

其中核心目标不是“把字段攒满”，而是：

> **确保进入 feature_history 的字段，真能支持“预测 T+1 最大概率涨停”这一最高目标。**

所以字段入池必须满足三条原则：

1. **有业务含义**
2. **有稳定来源**
3. **可长期落库、可回放、可训练**

---

## 3. 总体设计原则

### 3.1 字段分层原则

FeatureHistory 字段分为五层：

1. **主键与样本定位层**
2. **候选池基础层**
3. **盘口强度层**
4. **题材加权层**
5. **模型推理与真值反馈层**

### 3.2 入池基本规则

- 能从上游标准表稳定提取的，直接入池。
- 上游暂时缺失但业务重要的，允许先保留字段位，按缺失策略处理。
- 不允许把“临时调试字段”“不可回放字段”“未来函数字段”直接入池。
- 所有字段必须能回答：
  - 来源是谁？
  - 生成逻辑是什么？
  - 缺失怎么办？
  - 缺失到什么程度必须阻断？

### 3.3 闸门原则

FeatureHistory 不是“能写一行就算成功”，而是要设质量闸门：

- **行级闸门**：单条记录关键字段过缺，则该条样本不可训练。
- **批级闸门**：某日全量关键字段缺失率过高，则该日不得进入训练集。
- **链路闸门**：若上游关键源表异常，则当天 Step7 只追加审计，不追加训练样本。

---

## 4. 字段入池映射总表

> 说明：  
> - “来源表”优先写最终稳定来源。  
> - “落地逻辑”写入 `feature_history.csv` 时的口径。  
> - “缺失策略”分为：`置空` / `置0` / `回退` / `阻断训练`。  
> - “闸门阈值”优先服务训练质量，不服务“勉强跑通”。

---

## 4.1 主键与样本定位层

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| trade_date | 样本交易日 | 主流程运行日 / 输出日期 | 预测对应的 T 日，统一写 `YYYYMMDD` | 不允许缺失 | 缺失即阻断整批 |
| ts_code | 股票代码 | candidate_pool / pred_source | 候选股票代码，统一用 ts_code | 不允许缺失 | 缺失即丢弃该行 |
| name | 股票名称 | candidate_pool / stock_basic | 对应 ts_code 的股票简称 | 可回退为空字符串 | 缺失率 > 20% 预警，> 50% 阻断整批训练 |
| sample_id | 样本唯一键 | 系统生成 | 推荐口径：`trade_date + "_" + ts_code` | 自动生成 | 缺失即阻断该行 |
| in_topn | 是否进入最终TopN | Step6 输出 | 若当日入选 TopN 记 1，否则 0 | 置0 | 缺失率 > 5% 预警 |
| rank_final | 最终排序名次 | Step6 输出 | 按最终 EV / 综合分排序后的名次 | 置空 | TopN 行缺失率 > 0 预警 |
| run_id | 本次运行批次ID | 系统生成 | 记录一次主链运行批次，便于审计 | 可置空 | 非训练阻断字段 |
| version_tag | 版本标签 | 配置 / 系统常量 | 记录当前引擎版本、契约版本 | 可置空 | 非训练阻断字段 |

---

## 4.2 候选池基础层

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| close | 当日收盘价 | `daily.csv` | 取 T 日收盘价 | 不允许关键样本缺失 | 缺失即该行不可训练 |
| pct_chg | 当日涨跌幅 | `daily.csv` | 取 T 日涨跌幅 | 可由 close / pre_close 回算，否则置空 | 缺失率 > 10% 预警 |
| amount | 成交额 | `daily.csv` | 取 T 日成交额 | 置空 | 缺失率 > 20% 预警 |
| vol | 成交量 | `daily.csv` | 取 T 日成交量 | 置空 | 缺失率 > 20% 预警 |
| turnover_rate | 换手率 | `daily_basic.csv` | 优先用标准换手率字段 | 置空，不得乱填0 | 缺失率 > 20% 预警，> 40% 阻断训练日 |
| volume_ratio | 量比 | 若有标准表则取标准表，否则待补 | 当前若无稳定来源则保留字段位 | 置空 | 非当前阻断字段 |
| circ_mv | 流通市值 | `daily_basic.csv` | 取 T 日流通市值 | 置空 | 缺失率 > 20% 预警 |
| total_mv | 总市值 | `daily_basic.csv` | 取 T 日总市值 | 置空 | 缺失率 > 20% 预警 |
| is_st | ST标记 | `namechange.csv` / stock_basic / 规则判定 | 名称含 ST/*ST 或规则识别记 1 | 回退规则判断 | 缺失率 > 5% 预警 |
| list_days | 上市天数 | `stock_basic.csv` + trade_date | 用 trade_date - list_date 计算 | 置空 | 缺失率 > 20% 预警 |
| board_type | 板块类型 | `stock_basic.csv` / ts_code规则 | 主板/创业板/科创板/北交所等 | 回退为规则判断 | 缺失率 > 10% 预警 |

---

## 4.3 涨停与盘口强度层

> 这一层是当前最关键、也是最容易失真的层。  
> `open_times`、`seal_amount` 等字段必须明确来源，不允许“想当然填0”。

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| limit_type | 涨停类型 | `limit_list_d.csv` | 一字板/自然板/连板等，按源表原义落地 | 置空 | 缺失率 > 30% 预警 |
| up_limit | 涨停价 | `limit_list_d.csv` / `stk_limit.csv` | 优先标准涨停价字段 | 可回退 `stk_limit`，否则置空 | TopN 行缺失率 > 10% 预警 |
| down_limit | 跌停价 | `limit_list_d.csv` / `stk_limit.csv` | 优先标准跌停价字段 | 可回退 `stk_limit`，否则置空 | 非关键阻断字段 |
| is_limit_up_pool | 是否进入涨停池 | `limit_list_d.csv` | 当日若在涨停池记 1，否则 0 | 置0 | 缺失率 > 5% 预警 |
| open_times | 炸板次数 | `limit_break_d.csv` 优先；无则其它稳定炸板源 | T 日炸板次数 | **禁止无依据置0**；若源表缺，则置空 | 缺失率 > 40% 预警，> 70% 阻断训练日 |
| seal_amount | 封单金额 | `limit_list_d.csv` 优先；若无则待补稳定源 | T 日封单金额 | **禁止无依据置0**；若源表缺，则置空 | 缺失率 > 40% 预警，> 70% 阻断训练日 |
| first_limit_time | 首次封板时间 | 涨停明细表（待稳定） | T 日首次封板时间转为数值化特征 | 置空 | 非当前阻断字段 |
| last_limit_time | 最后封板时间 | 涨停明细表（待稳定） | T 日最后封板时间转为数值化特征 | 置空 | 非当前阻断字段 |
| limit_open_count | 开板次数 | `limit_break_d.csv` / 明细表 | 若与 open_times 区分存在，则单独保留 | 置空 | 非当前阻断字段 |
| limit_strength_raw | 原始涨停强度分 | Step3 | 由 Step3 计算出的未标准化强度 | 若 Step3 未产出则置空 | 缺失率 > 10% 预警 |
| StrengthScore | 最终强度分 | Step3 | Step3 的最终标准输出值 | 不允许关键样本缺失 | 缺失率 > 10% 预警，> 30% 阻断训练日 |

---

## 4.4 题材与市场热度层

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| theme_main | 主题材 | `hot_boards.csv` / 题材映射结果 | 记录命中的主题材名称 | 置空 | 非训练阻断字段 |
| theme_count | 命中题材数 | 题材映射结果 | 一只股票命中的题材数量 | 置0 | 缺失率 > 20% 预警 |
| theme_heat | 题材热度值 | `hot_boards.csv` | 对应主题材热度分或聚合热度 | 置空 | 缺失率 > 30% 预警 |
| board_rank | 题材/板块热度排名 | `hot_boards.csv` | 热门板块中的排序名次 | 置空 | 非关键阻断字段 |
| lhb_flag | 龙虎榜标记 | `top_list.csv` | 当日上榜记 1，否则 0 | 置0 | 缺失率 > 10% 预警 |
| lhb_net_buy | 龙虎榜净买额 | `top_list.csv` | 可聚合则取净买额 | 置空 | 非关键阻断字段 |
| north_money_flag | 北向关注标记 | `moneyflow_hsgt.csv` / 规则派生 | 若能映射则记 1，否则 0 | 置0 | 非关键阻断字段 |
| ThemeBoost | 题材加权得分 | Step4 | Step4 的最终输出值 | 不允许关键样本缺失 | 缺失率 > 10% 预警，> 30% 阻断训练日 |

---

## 4.5 模型推理层

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| Probability | 最终涨停概率 | Step5 | Step5 最终融合概率输出 | 不允许关键样本缺失 | 缺失率 > 5% 预警，> 20% 阻断训练日 |
| prob_lr | LR模型概率 | Step5 | 若使用 LR 则记录原始概率 | 置空 | 非当前阻断字段 |
| prob_lgbm | LGBM模型概率 | Step5 | 若使用 LGBM 则记录原始概率 | 置空 | 非当前阻断字段 |
| prob_rule | 规则层概率/先验分 | Step5 / Rule层 | 若存在规则先验则保留 | 置空 | 非当前阻断字段 |
| _prob_src | 概率来源说明 | Step5 | 记录 `lr` / `lgbm` / `blend` / `fallback_rule` 等 | 置空但强烈建议保留 | TopN 行缺失率 > 20% 预警 |
| model_version | 推理模型版本 | Step5 / models meta | 当前推理所用模型版本 | 置空 | 非训练阻断字段 |
| feature_level | 特征成熟层级 | 系统计算 | 如 L1/L2/L3 | 置空 | 非当前阻断字段 |
| score_raw | 原始综合分 | Step5 / Step6 | 若有 pre-rank score 则保留 | 置空 | 非当前阻断字段 |

---

## 4.6 真值反馈与学习层

| 字段英文名 | 中文含义 | 来源表 | 落地逻辑 | 缺失策略 | 闸门阈值 |
|---|---|---|---|---|---|
| y_limit_hit | 次日是否涨停命中 | T+1 真值表 / `limit_list_d.csv` | 若 T+1 涨停命中记 1，否则 0 | 样本未成熟时置空 | 未成熟样本不得训练 |
| y_next_ret | 次日收益率 | T+1 日行情 | 可记录 T+1 收盘相对 T 日收盘收益 | 样本未成熟时置空 | 非当前主标签阻断字段 |
| is_sample_mature | 样本是否成熟 | Step7 判定 | 到了真值可得日期记 1 | 不允许缺失 | 缺失率 > 0 预警 |
| learnable_flag | 是否可进入训练 | Step7 闸门 | 满足关键字段+样本成熟+不过闸门记 1 | 自动生成 | 缺失即按 0 处理 |
| reject_reason | 剔除原因 | Step7 | 记录为何不可训练：字段缺失/样本未成熟/真值缺失等 | 置空 | 非训练阻断字段 |
| hit_rank_bucket | 命中分层桶 | Step7 | 命中样本所在名次桶 | 置空 | 非当前阻断字段 |
| created_at | 入池时间戳 | 系统生成 | 写入 feature_history 的时间 | 自动生成 | 非训练阻断字段 |

---

## 5. 当前必须锁死的关键字段

当前阶段不是所有字段都同等重要。  
必须优先锁死、优先保障、优先修通的字段如下：

### 5.1 一级关键字段（没有就不能训练）

- `trade_date`
- `ts_code`
- `close`
- `turnover_rate`
- `StrengthScore`
- `ThemeBoost`
- `Probability`
- `_prob_src`
- `is_sample_mature`
- `y_limit_hit`

### 5.2 二级关键字段（高缺失会显著污染训练）

- `open_times`
- `seal_amount`
- `limit_type`
- `up_limit`
- `is_limit_up_pool`
- `theme_heat`
- `lhb_flag`

### 5.3 三级增强字段（可后续逐步补强）

- `first_limit_time`
- `last_limit_time`
- `lhb_net_buy`
- `north_money_flag`
- `board_rank`
- `volume_ratio`

---

## 6. 缺失策略统一口径

这是全链路最容易失控的地方，必须统一。

### 6.1 可以置0的字段

仅当业务语义明确“没有就是0”时，才允许置0，例如：

- `in_topn`
- `is_limit_up_pool`
- `lhb_flag`
- `theme_count`
- `north_money_flag`
- `learnable_flag`

### 6.2 不允许随便置0的字段

以下字段若来源缺失，**只能置空，不得伪造0**：

- `open_times`
- `seal_amount`
- `turnover_rate`
- `StrengthScore`
- `ThemeBoost`
- `Probability`
- `up_limit`
- `down_limit`
- `theme_heat`

原因很简单：

> 这些字段的 0 不是“无”，而是“一个有含义的真实数值”。  
> 一旦伪造 0，模型会把“缺失”误学成“极弱特征”。

### 6.3 回退策略

允许回退的字段示例：

- `up_limit`：可从 `stk_limit.csv` 回退
- `down_limit`：可从 `stk_limit.csv` 回退
- `is_st`：可从名称规则回退
- `board_type`：可从代码规则回退
- `name`：可从 stock_basic 回退

### 6.4 阻断策略

以下情况必须阻断训练：

1. `trade_date` / `ts_code` 缺失
2. `Probability` 缺失
3. `StrengthScore` 缺失
4. `ThemeBoost` 缺失
5. `y_limit_hit` 未成熟却混入训练
6. `open_times`、`seal_amount` 在整批中大面积失真并突破阈值

---

## 7. FeatureHistory 行级闸门

单条样本进入训练前，建议执行如下行级闸门：

### 7.1 必过条件

若以下任一不满足，则 `learnable_flag = 0`：

- `trade_date` 非空
- `ts_code` 非空
- `is_sample_mature = 1`
- `StrengthScore` 非空
- `ThemeBoost` 非空
- `Probability` 非空
- `y_limit_hit` 非空

### 7.2 强度字段条件

若以下字段全部缺失，则样本不应直接训练：

- `turnover_rate`
- `open_times`
- `seal_amount`

理由：

> 这三项共同决定“当日强弱与可持续性”的关键横截面。  
> 如果这层全空，训练样本会退化成只有静态价格壳。

### 7.3 记录 reject_reason

拒绝进入训练时必须写原因，例如：

- `missing_trade_date`
- `missing_ts_code`
- `sample_not_mature`
- `missing_strengthscore`
- `missing_themeboost`
- `missing_probability`
- `missing_label`
- `missing_microstructure_cluster`

---

## 8. 批级闸门（按交易日）

建议 Step7 对每个 `trade_date` 做批级检查：

### 8.1 阻断训练日的条件

满足任一条件，则该日样本不进入正式训练集：

1. `StrengthScore` 缺失率 > 30%
2. `ThemeBoost` 缺失率 > 30%
3. `Probability` 缺失率 > 20%
4. `turnover_rate` 缺失率 > 40%
5. `open_times` 缺失率 > 70%
6. `seal_amount` 缺失率 > 70%
7. 当日样本总数 < 最低训练门槛

### 8.2 样本总数门槛

建议三档：

- **Level 1**：可学习最低门槛：`>= 20`
- **Level 2**：基本可信门槛：`>= 50`
- **Level 3**：较稳健门槛：`>= 100`

说明：

- `< 20`：只做审计，不训练
- `20 ~ 49`：弱训练，低权重
- `50 ~ 99`：正常训练
- `>= 100`：稳健训练，可作为主样本批

---

## 9. 当前已知问题与本表对应关系

结合当前主线，已知最关键问题如下：

### 9.1 `limit_break_d.csv` 为空

影响字段：

- `open_times`
- `limit_open_count`

处理原则：

- 当前允许字段保留，但不得伪造
- 必须在数据仓库侧补齐快照生成逻辑

### 9.2 `limit_list_d.csv` 字段不完整

当前已知会影响：

- `limit_type`
- `up_limit`
- `down_limit`
- `seal_amount`

处理原则：

- `up_limit/down_limit` 可尝试从 `stk_limit.csv` 回退
- `seal_amount` 无可靠源时只能置空
- 若缺失过高，则当天不得进入正式训练集

### 9.3 feature_history 已有核心字段，但微观盘口仍薄

当前可认为已经部分具备训练骨架的字段：

- `StrengthScore`
- `turnover_rate`
- `ThemeBoost`
- `Probability`
- `_prob_src`

但尚未形成专业机构级样本质量的关键缺口仍是：

- `open_times`
- `seal_amount`
- 更细的封板时序字段

---

## 10. 建议的落地顺序

这个顺序不能乱。

### 第一阶段：锁死字段契约

先按本表把字段定义统一下来，不再漂移。

目标：

- 字段名统一
- 来源表统一
- 缺失策略统一
- 闸门阈值统一

### 第二阶段：修数据仓库源表

优先进入 `a-share-top3-data` 修复：

- `limit_break_d.csv`
- `limit_list_d.csv`
- 必要时补充其它可提供 `seal_amount/open_times` 的源表链路

目标：

- 让关键微观盘口字段真正可得
- 不再靠“空值硬撑”

### 第三阶段：改 a-top10 入池逻辑

按本表更新：

- Step3 写 StrengthScore
- Step4 写 ThemeBoost
- Step5 写 Probability / `_prob_src`
- Step7 写成熟度、标签、reject_reason、learnable_flag

目标：

- `feature_history.csv` 从“流水记录”升级为“训练样本契约表”

### 第四阶段：加质量闸门

把本表中的行级/批级闸门写进 Step7。

目标：

- 不让脏样本混入训练
- 不让“勉强跑通”毁掉长期模型

### 第五阶段：再谈增强字段扩容

等一级、二级关键字段稳定后，再扩：

- 首板/末板时序
- 龙虎榜细项
- 北向流入
- 板块相对热度排名
- 更多盘口行为特征

---

## 11. 对开发执行的硬性要求

后续任何开发，必须遵守以下规则：

1. **新增 FeatureHistory 字段，必须先更新本表。**
2. **修改字段名，必须先改契约，再改代码。**
3. **来源表变化，必须更新本表。**
4. **缺失策略变化，必须更新本表。**
5. **闸门阈值调整，必须记版本。**

否则系统会重新进入“字段混乱、训练漂移、开发失控”的老问题。

---

## 12. V1 版本结论

《FeatureHistory 字段入池映射表 V1》的作用，不是为了“写个说明文档”，而是为了完成三件事：

### 12.1 先把样本口径锁死

让后续所有开发，都围绕同一份字段契约推进。

### 12.2 把当前主线问题显性化

明确告诉系统：

> 现在不是 Step5 模型公式最致命，  
> 而是 **FeatureHistory 的微观盘口字段链路还不够硬**。

### 12.3 为 V2 / V3 升级留好锚点

后续若进入更高阶版本，可在本表基础上继续扩展：

- V2：增加时序盘口字段组
- V3：增加市场环境与相对强弱字段组
- V4：增加跨日序列与样本成熟度权重机制

---

## 13. V1 后续直接动作

本文件落定后，下一步建议直接进入：

### 动作 A：输出《FeatureHistory 字段入池映射表 V1》对应的代码改造清单

按文件拆分：

- `a_top10/steps/step3_strength_score.py`
- `a_top10/steps/step4_theme_boost.py`
- `a_top10/steps/step5_ml_probability.py`
- `a_top10/steps/step7_self_learning.py`
- `a_top10/io/writers.py`
- 必要的 schema / config 文件

### 动作 B：输出《FeatureHistory 质量闸门 V1》

把：
- 行级闸门
- 批级闸门
- 样本成熟规则
- 训练放行规则

单独整理成下一份锚点文档。

### 动作 C：进入数据仓库修复主线

去 `a-share-top3-data` 修：
- `limit_break_d.csv`
- `limit_list_d.csv`
- 相关快照生成逻辑

---

**文档版本**：V1  
**文档定位**：FeatureHistory 字段契约锚点  
**状态**：可立即作为后续开发与验收依据
