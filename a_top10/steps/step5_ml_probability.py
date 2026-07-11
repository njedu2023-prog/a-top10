#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 : ML Probability — Top10 V3

定位：
- Step5 不再是 V2 的“prob_ml / prob_final 兼容层”，而是 V3 的主概率引擎。
- 明确输出：
    - prob_lr
    - prob_lgbm
    - prob_rule
    - Probability
    - rank_score
    - probability_is_calibrated
    - probability_semantics
    - model_contract_status
    - model_contract_reason
    - _prob_src
- 其中：
    - Probability = 最终主排序概率轴
    - _prob_src = 最终概率来源，必须可追踪

兼容目标：
- 保留训练接口：
    - train_step5_lr
    - train_step5_lgbm
    - train_step5_models
- 保留推理入口：
    - run_step5(theme_df, s=None)
    - run(theme_df, s=None)

V3 原则：
- Step5 内部计算允许对缺失特征做数值兜底，但不得污染上游契约字段
- 若 ML 模型不可用，则必须显式回退到 prob_rule，并写明 _prob_src
- prob_rule 仅是未校准排序分；兼容列 Probability 不代表已校准概率
- 模型推理前必须通过有序特征契约，拒绝原因和推理错误必须可审计
- 若 Probability 无法得到，则样本不应被当作正常概率样本消费
- Step5 不再越权写 feature_history.csv；正式样本落库由 writers.py / Step7 负责
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from a_top10.intraday_features import ML_INTRADAY_FEATURES
from a_top10.config import is_a_share_trading_day, prev_a_share_trading_day

try:
    import joblib
except Exception:
    joblib = None

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


# =========================================================
# Config
# =========================================================

CORE_FEATURES = [
    "StrengthScore",
    "ThemeBoost",
    "seal_amount",
    "open_times",
    "turnover_rate",
]

FEATURES = CORE_FEATURES + [
    "limitup_quality_score",
    "intraday_quality_score",
    "intraday_soft_risk_score",
    "intraday_hard_risk_flag",
    "intraday_risk_score",
    "late_withdraw_score",
    "reseal_score",
    "open_board_count",
    "open_board_risk_score",
    "auction_strength_score",
    "auction_real_volume_score",
    "seal_stability_score",
    "intraday_confidence_score",
    "strength_plus_score",
    "intraday_available",
    "auction_available",
]

# Existing bare estimators remain supported through sklearn metadata. New
# artifacts may additionally carry this manifest, either as an estimator
# attribute or as {"model": estimator, "manifest": {...}}.
MODEL_CONTRACT_SCHEMA_VERSION = 1
MODEL_FEATURE_SCHEMA_VERSION = "step5_features_v1"
CORE_FEATURE_SCHEMA_VERSION = "step5_core_features_v1"
MODEL_MANIFEST_ATTR = "step5_model_manifest_"
FEATURE_CONTRACT = tuple(FEATURES)
CORE_FEATURE_CONTRACT = tuple(CORE_FEATURES)

VALID_RUN_MODES = {"replay", "train", "auto_daily"}


# =========================================================
# Basic utils
# =========================================================

def _ensure_df(x: Any) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _first_existing_col(df: pd.DataFrame, cands: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in cands:
        hit = lower_map.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def _get_ts_code_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["ts_code", "code", "TS_CODE", "证券代码", "股票代码"])


def _get_name_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_col(df, ["name", "stock_name", "名称", "股票简称", "证券名称"])


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_df(df)
    ts_col = _get_ts_code_col(df)
    if ts_col and ts_col != "ts_code":
        df["ts_code"] = df[ts_col].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str).str.strip()
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).str.replace(r"\.0+$", "", regex=True).str.strip()
    if "verify_date" in df.columns:
        df["verify_date"] = df["verify_date"].astype(str).str.replace(r"\.0+$", "", regex=True).str.strip()
    return df


def _to_numeric_nullable(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.astype("float64").clip(0.0, 1.0)
    arr = np.asarray(s, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    return pd.Series(arr, dtype="float64")


def _utc_now_iso() -> str:
    try:
        return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _guess_trade_date(df: pd.DataFrame) -> str:
    df = _ensure_df(df)
    for c in ["trade_date", "TradeDate", "日期", "交易日期"]:
        if c in df.columns and len(df) > 0:
            v = str(df[c].iloc[0]).strip()
            if re.match(r"^\d{8}$", v):
                return v
    env_td = os.getenv("TRADE_DATE", "").strip()
    if re.match(r"^\d{8}$", env_td):
        try:
            if is_a_share_trading_day(env_td):
                return env_td
            return prev_a_share_trading_day(env_td)
        except Exception:
            pass
    return prev_a_share_trading_day(pd.Timestamp.today().strftime("%Y%m%d"))


def _get_outputs_dir(s=None) -> Path:
    try:
        io_obj = getattr(s, "io", None)
        out = getattr(io_obj, "outputs_dir", None)
        if out:
            return Path(str(out))
    except Exception:
        pass
    return Path("outputs")


def _get_ml_cfg(s=None) -> Dict[str, Any]:
    cfg = {
        "model": "auto",                  # auto | logistic | lightgbm
        "enable_rule": True,
        "enable_ml": True,
        "fusion_mode": "ml_first",       # ml_first | weighted
        "fallback_to_rule": True,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "rule_weight": 0.30,
        "ml_weight": 0.70,
    }
    try:
        ml = getattr(s, "ml", None)
        if ml is None:
            return cfg
        for k in list(cfg.keys()):
            if hasattr(ml, k):
                cfg[k] = getattr(ml, k)
    except Exception:
        pass

    cfg["model"] = str(cfg.get("model", "auto")).lower()
    cfg["fusion_mode"] = str(cfg.get("fusion_mode", "ml_first")).lower()
    cfg["enable_rule"] = bool(cfg.get("enable_rule", True))
    cfg["enable_ml"] = bool(cfg.get("enable_ml", True))
    cfg["fallback_to_rule"] = bool(cfg.get("fallback_to_rule", True))
    cfg["clip_min"] = float(cfg.get("clip_min", 0.0))
    cfg["clip_max"] = float(cfg.get("clip_max", 1.0))
    cfg["rule_weight"] = float(cfg.get("rule_weight", 0.30))
    cfg["ml_weight"] = float(cfg.get("ml_weight", 0.70))
    return cfg


# =========================================================
# Run mode / training policy
# =========================================================

@dataclass
class Step5TrainingPolicy:
    run_mode: str
    allow_model_update_raw: str
    allow_model_update: bool
    min_level1_samples: int
    min_train_samples: int
    min_stable_samples: int
    min_positive_samples: int
    min_feature_coverage: float


def _resolve_run_mode() -> str:
    mode = os.getenv("A_TOP10_RUN_MODE", os.getenv("TOP10_RUN_MODE", "auto_daily")).strip().lower()
    if mode not in VALID_RUN_MODES:
        return "auto_daily"
    return mode


def _resolve_allow_model_update(run_mode: str) -> Tuple[str, bool]:
    raw = os.getenv(
        "A_TOP10_ALLOW_MODEL_UPDATE",
        os.getenv("TOP10_ALLOW_MODEL_UPDATE", "auto"),
    ).strip().lower()

    if raw in {"1", "true", "yes", "on"}:
        return raw, True
    if raw in {"0", "false", "no", "off"}:
        return raw, False

    if run_mode == "replay":
        return "auto", False
    if run_mode == "train":
        return "auto", True
    return "auto", True


def _policy_value_from_settings(s: Any, *names: str, default: Any) -> Any:
    for obj_name in ["training", "ml"]:
        try:
            obj = getattr(s, obj_name, None)
            if obj is None:
                continue
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    if v is not None:
                        return v
        except Exception:
            pass
    return default


def _get_training_policy(s=None) -> Step5TrainingPolicy:
    run_mode = _resolve_run_mode()
    allow_raw, allow_update = _resolve_allow_model_update(run_mode)

    min_level1_samples = int(_policy_value_from_settings(s, "min_level1_samples", default=80))
    min_train_samples = int(_policy_value_from_settings(s, "min_train_samples", default=120))
    min_stable_samples = int(_policy_value_from_settings(s, "min_stable_samples", default=150))
    min_positive_samples = int(_policy_value_from_settings(s, "min_positive_samples", default=12))
    min_feature_coverage = float(_policy_value_from_settings(s, "min_feature_coverage", default=0.85))

    return Step5TrainingPolicy(
        run_mode=run_mode,
        allow_model_update_raw=allow_raw,
        allow_model_update=allow_update,
        min_level1_samples=min_level1_samples,
        min_train_samples=min_train_samples,
        min_stable_samples=min_stable_samples,
        min_positive_samples=min_positive_samples,
        min_feature_coverage=min_feature_coverage,
    )


def _classify_training_level(n_samples: int, policy: Step5TrainingPolicy) -> str:
    if n_samples < policy.min_level1_samples:
        return "below_level1"
    if n_samples < policy.min_train_samples:
        return "level1"
    if n_samples < policy.min_stable_samples:
        return "level2"
    return "level3"


def _can_formal_update(level: str, policy: Step5TrainingPolicy) -> bool:
    if not policy.allow_model_update:
        return False
    return level in {"level2", "level3"}


# =========================================================
# Step3 feature backfill
# =========================================================

def _backfill_features_from_step3(raw_df: pd.DataFrame, trade_date: str, outputs_dir: Path) -> pd.DataFrame:
    """
    允许从 Step3 产物回填核心特征，但只在当前 df 缺列或几乎全空/全零时回填。
    """
    raw_df = _normalize_id_columns(raw_df)
    if raw_df.empty or "ts_code" not in raw_df.columns:
        return raw_df

    need_cols = ["StrengthScore", "seal_amount", "open_times", "turnover_rate"]
    missing = []

    for c in need_cols:
        if c not in raw_df.columns:
            missing.append(c)
            continue

        vals = _to_numeric_nullable(raw_df[c])
        nonnull_ratio = float(vals.notna().mean()) if len(vals) else 0.0
        nonzero_ratio = float((vals.fillna(0.0) != 0.0).mean()) if len(vals) else 0.0

        if nonnull_ratio < 0.01 or nonzero_ratio < 0.01:
            missing.append(c)

    if not missing:
        return raw_df

    candidates = [
        outputs_dir / f"step3_strength_{trade_date}.csv",
        outputs_dir / "step3_strength.csv",
        Path("outputs") / f"step3_strength_{trade_date}.csv",
        Path("outputs") / "step3_strength.csv",
    ]
    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if src is None:
        return raw_df

    try:
        step3_df = pd.read_csv(src)
    except Exception:
        return raw_df

    step3_df = _normalize_id_columns(step3_df)
    if step3_df.empty or "ts_code" not in step3_df.columns:
        return raw_df

    use_cols = ["ts_code"] + [c for c in need_cols if c in step3_df.columns]
    if len(use_cols) <= 1:
        return raw_df

    merged = raw_df.merge(step3_df[use_cols], on="ts_code", how="left", suffixes=("", "_s3"))

    for c in need_cols:
        s3 = f"{c}_s3"
        if c in merged.columns and s3 in merged.columns:
            cur = _to_numeric_nullable(merged[c])
            ext = _to_numeric_nullable(merged[s3])

            cur_bad = cur.isna() | (cur.fillna(0.0) == 0.0)
            merged[c] = cur.where(~cur_bad, ext)
            merged.drop(columns=[s3], inplace=True, errors="ignore")
        elif c not in merged.columns and s3 in merged.columns:
            merged[c] = _to_numeric_nullable(merged[s3])
            merged.drop(columns=[s3], inplace=True, errors="ignore")

    return merged


# =========================================================
# Inference feature prep
# =========================================================

def _ensure_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅为 Step5 内部推理准备数值特征。
    不对外伪造契约字段。
    """
    df = _ensure_df(df)
    defaults = {
        "limitup_quality_score": 0.55,
        "intraday_quality_score": 0.55,
        "intraday_soft_risk_score": 0.00,
        "intraday_hard_risk_flag": 0.00,
        "intraday_risk_score": 0.00,
        "late_withdraw_score": 0.00,
        "reseal_score": 0.50,
        "open_board_count": 0.00,
        "open_board_risk_score": 0.00,
        "auction_strength_score": 0.50,
        "auction_real_volume_score": 0.50,
        "seal_stability_score": 0.50,
        "intraday_confidence_score": 0.00,
        "strength_plus_score": np.nan,
        "intraday_available": 0.00,
        "auction_available": 0.00,
    }
    for c in FEATURES:
        if c not in df.columns:
            df[c] = defaults.get(c, np.nan)

    feat = df.copy()
    for c in FEATURES:
        default = defaults.get(c, 0.0)
        if c == "strength_plus_score":
            if "StrengthScore" in feat.columns:
                default_sr = pd.to_numeric(feat["StrengthScore"], errors="coerce").fillna(0.0)
            else:
                default_sr = pd.Series([0.0] * len(feat), index=feat.index, dtype="float64")
            feat[c] = _to_numeric_nullable(feat[c]).fillna(default_sr)
        else:
            feat[c] = _to_numeric_nullable(feat[c]).fillna(float(default))

    return feat


# =========================================================
# Model IO
# =========================================================

@dataclass
class Step5ModelPaths:
    base: Path
    lr_path: Path
    hgb_path: Path
    lgbm_path: Path


@dataclass(frozen=True)
class ModelArtifactError:
    status: str
    reason: str


@dataclass(frozen=True)
class ModelContractCheck:
    status: str
    reason: str
    expected_feature_count: int
    actual_feature_count: Optional[int]
    expected_features: Tuple[str, ...]
    actual_features: Tuple[str, ...]
    manifest_schema_version: Optional[str]
    feature_schema_version: Optional[str]
    probability_is_calibrated: bool

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


@dataclass
class ModelPredictionResult:
    values: pd.Series
    contract: ModelContractCheck
    prediction_status: str
    prediction_error: str = ""


def _empty_model_contract(
    status: str,
    reason: str,
    expected_features: Sequence[str] = FEATURE_CONTRACT,
) -> ModelContractCheck:
    expected = tuple(str(x) for x in expected_features)
    return ModelContractCheck(
        status=status,
        reason=reason,
        expected_feature_count=len(expected),
        actual_feature_count=None,
        expected_features=expected,
        actual_features=(),
        manifest_schema_version=None,
        feature_schema_version=None,
        probability_is_calibrated=False,
    )


def _model_manifest(model: Any) -> Dict[str, Any]:
    if isinstance(model, dict):
        raw = model.get("manifest", model.get("model_manifest", {}))
        return dict(raw) if isinstance(raw, dict) else {}

    for attr in [MODEL_MANIFEST_ATTR, "model_manifest_", "manifest_"]:
        raw = getattr(model, attr, None)
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def _unwrap_model(model: Any) -> Any:
    if isinstance(model, dict) and "model" in model:
        return model.get("model")
    return model


def _manifest_value(manifest: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in manifest:
            return manifest.get(key)
    return None


def _normalize_feature_contract(raw: Any) -> Tuple[str, ...]:
    if raw is None or isinstance(raw, (str, bytes)):
        return ()
    try:
        return tuple(str(x) for x in list(raw))
    except Exception:
        return ()


def _feature_schema_version(features: Sequence[str]) -> str:
    contract = tuple(str(x) for x in features)
    if contract == CORE_FEATURE_CONTRACT:
        return CORE_FEATURE_SCHEMA_VERSION
    return MODEL_FEATURE_SCHEMA_VERSION


def _model_feature_contract(model: Any) -> Tuple[str, ...]:
    """Use only a versioned supported manifest to select a reduced contract."""
    manifest_features = _normalize_feature_contract(
        _manifest_value(_model_manifest(model), "features", "feature_names", "feature_names_in")
    )
    if manifest_features in {CORE_FEATURE_CONTRACT, FEATURE_CONTRACT}:
        return manifest_features
    # Bare legacy models are intentionally checked against the full V3
    # contract, so an old anonymous five-column artifact cannot be guessed.
    return FEATURE_CONTRACT


def _feature_mismatch_reason(expected: Sequence[str], actual: Sequence[str]) -> str:
    expected_list = [str(x) for x in expected]
    actual_list = [str(x) for x in actual]
    missing = [x for x in expected_list if x not in actual_list]
    unexpected = [x for x in actual_list if x not in expected_list]
    order_mismatch = not missing and not unexpected and expected_list != actual_list
    return (
        f"expected_features={expected_list}; actual_features={actual_list}; "
        f"missing_features={missing}; unexpected_features={unexpected}; "
        f"order_mismatch={str(order_mismatch).lower()}"
    )


def _validate_model_contract(
    model: Any,
    expected_features: Sequence[str] = FEATURE_CONTRACT,
) -> ModelContractCheck:
    expected = tuple(str(x) for x in expected_features)
    if isinstance(model, ModelArtifactError):
        return _empty_model_contract(model.status, model.reason, expected)

    estimator = _unwrap_model(model)
    manifest = _model_manifest(model)

    if estimator is None:
        return _empty_model_contract("model_missing", "model artifact is missing", expected)

    manifest_schema_raw = _manifest_value(manifest, "contract_schema_version", "schema_version")
    manifest_schema = str(manifest_schema_raw) if manifest_schema_raw is not None else None
    feature_schema_raw = _manifest_value(manifest, "feature_schema_version", "features_schema_version")
    feature_schema = str(feature_schema_raw) if feature_schema_raw is not None else None
    calibrated = bool(manifest.get("probability_is_calibrated", False))

    if manifest_schema is not None and manifest_schema != str(MODEL_CONTRACT_SCHEMA_VERSION):
        return ModelContractCheck(
            status="manifest_schema_unsupported",
            reason=(
                f"expected_contract_schema_version={MODEL_CONTRACT_SCHEMA_VERSION}; "
                f"actual_contract_schema_version={manifest_schema}"
            ),
            expected_feature_count=len(expected),
            actual_feature_count=None,
            expected_features=expected,
            actual_features=(),
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    expected_feature_schema = _feature_schema_version(expected)
    if feature_schema is not None and feature_schema != expected_feature_schema:
        return ModelContractCheck(
            status="feature_schema_mismatch",
            reason=(
                f"expected_feature_schema_version={expected_feature_schema}; "
                f"actual_feature_schema_version={feature_schema}"
            ),
            expected_feature_count=len(expected),
            actual_feature_count=None,
            expected_features=expected,
            actual_features=(),
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    manifest_features = _normalize_feature_contract(
        _manifest_value(manifest, "features", "feature_names", "feature_names_in")
    )
    if manifest_features and manifest_features != expected:
        return ModelContractCheck(
            status="manifest_feature_mismatch",
            reason=_feature_mismatch_reason(expected, manifest_features),
            expected_feature_count=len(expected),
            actual_feature_count=len(manifest_features),
            expected_features=expected,
            actual_features=manifest_features,
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    actual_count: Optional[int] = None
    raw_count = getattr(estimator, "n_features_in_", None)
    if raw_count is not None:
        try:
            actual_count = int(raw_count)
        except (TypeError, ValueError):
            return ModelContractCheck(
                status="invalid_feature_count_metadata",
                reason=f"n_features_in_ is not an integer: {raw_count!r}",
                expected_feature_count=len(expected),
                actual_feature_count=None,
                expected_features=expected,
                actual_features=(),
                manifest_schema_version=manifest_schema,
                feature_schema_version=feature_schema,
                probability_is_calibrated=False,
            )

    actual_features = _normalize_feature_contract(getattr(estimator, "feature_names_in_", None))
    if actual_count is None and actual_features:
        actual_count = len(actual_features)

    if actual_count is not None and actual_count != len(expected):
        return ModelContractCheck(
            status="feature_count_mismatch",
            reason=f"expected_feature_count={len(expected)}; actual_feature_count={actual_count}",
            expected_feature_count=len(expected),
            actual_feature_count=actual_count,
            expected_features=expected,
            actual_features=actual_features,
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    if actual_features and actual_features != expected:
        return ModelContractCheck(
            status="feature_names_mismatch",
            reason=_feature_mismatch_reason(expected, actual_features),
            expected_feature_count=len(expected),
            actual_feature_count=actual_count,
            expected_features=expected,
            actual_features=actual_features,
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    if actual_count is None and not actual_features and not manifest_features:
        return ModelContractCheck(
            status="feature_metadata_missing",
            reason="model has no n_features_in_, feature_names_in_, or manifest feature contract",
            expected_feature_count=len(expected),
            actual_feature_count=None,
            expected_features=expected,
            actual_features=(),
            manifest_schema_version=manifest_schema,
            feature_schema_version=feature_schema,
            probability_is_calibrated=False,
        )

    return ModelContractCheck(
        status="valid",
        reason="ordered feature contract validated",
        expected_feature_count=len(expected),
        actual_feature_count=actual_count if actual_count is not None else len(manifest_features),
        expected_features=expected,
        actual_features=actual_features or manifest_features,
        manifest_schema_version=manifest_schema,
        feature_schema_version=feature_schema,
        probability_is_calibrated=calibrated,
    )


def _build_model_manifest(
    model_kind: str,
    *,
    features: Sequence[str] = FEATURE_CONTRACT,
    probability_is_calibrated: bool = False,
    calibration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    contract = tuple(str(x) for x in features)
    manifest = {
        "contract_schema_version": MODEL_CONTRACT_SCHEMA_VERSION,
        "feature_schema_version": _feature_schema_version(contract),
        "features": list(contract),
        "model_kind": str(model_kind),
        "probability_is_calibrated": bool(probability_is_calibrated),
        "created_at_utc": _utc_now_iso(),
    }
    if calibration:
        manifest["calibration"] = dict(calibration)
    return manifest


def _attach_model_manifest(
    model: Any,
    model_kind: str,
    *,
    features: Sequence[str] = FEATURE_CONTRACT,
    probability_is_calibrated: bool = False,
    calibration: Optional[Dict[str, Any]] = None,
) -> Any:
    try:
        setattr(
            model,
            MODEL_MANIFEST_ATTR,
            _build_model_manifest(
                model_kind,
                features=features,
                probability_is_calibrated=probability_is_calibrated,
                calibration=calibration,
            ),
        )
    except Exception:
        pass
    return model


def _get_model_paths(s=None) -> Step5ModelPaths:
    base = Path("models")
    try:
        dr = getattr(s, "data_repo", None)
        if dr is not None:
            for attr in ["models_dir", "model_dir"]:
                if hasattr(dr, attr):
                    cand = getattr(dr, attr)
                    if cand:
                        base = Path(str(cand))
                        break
    except Exception:
        pass

    _ensure_dir(base)
    return Step5ModelPaths(
        base=base,
        lr_path=base / "step5_lr.joblib",
        hgb_path=base / "step5_hgb.joblib",
        lgbm_path=base / "step5_lgbm.joblib",
    )


def _load_joblib(path: Path):
    if not path.exists():
        return None
    if joblib is None:
        return ModelArtifactError(
            status="joblib_unavailable",
            reason=f"cannot load model artifact {path}: joblib is unavailable",
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        return ModelArtifactError(
            status="model_load_error",
            reason=f"cannot load model artifact {path}: {type(exc).__name__}: {exc}",
        )


def _save_joblib(obj: Any, path: Path) -> None:
    if joblib is None:
        return
    try:
        joblib.dump(obj, path)
        manifest = _model_manifest(obj)
        if manifest:
            manifest_path = path.with_suffix(".manifest.json")
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        pass


def load_lr(s=None):
    return _load_joblib(_get_model_paths(s).lr_path)


def load_hgb(s=None):
    return _load_joblib(_get_model_paths(s).hgb_path)


def load_lgbm(s=None):
    return _load_joblib(_get_model_paths(s).lgbm_path)


# =========================================================
# Training dataset
# =========================================================

def _load_feature_history(s) -> pd.DataFrame:
    outputs_dir = _get_outputs_dir(s)
    fp = outputs_dir / "learning" / "feature_history.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fp)
    except Exception:
        try:
            return pd.read_csv(fp, encoding="gbk")
        except Exception:
            return pd.DataFrame()


def _sample_feature_coverage(
    df: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_CONTRACT,
) -> float:
    if df is None or df.empty:
        return 0.0

    coverages = []
    for c in feature_names:
        if c not in df.columns:
            coverages.append(0.0)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        coverages.append(float(s.notna().mean()))
    return float(np.mean(coverages)) if coverages else 0.0


def _select_training_feature_contract(
    df: pd.DataFrame,
    min_feature_coverage: float,
) -> Tuple[Tuple[str, ...], Dict[str, float | str]]:
    core_coverage = _sample_feature_coverage(df, CORE_FEATURE_CONTRACT)
    enhanced_coverage = _sample_feature_coverage(df, FEATURE_CONTRACT)
    intraday_rate = float(
        pd.to_numeric(
            df.get("intraday_available", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0).gt(0).mean()
    ) if len(df) else 0.0
    auction_rate = float(
        pd.to_numeric(
            df.get("auction_available", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0).gt(0).mean()
    ) if len(df) else 0.0
    use_enhanced = (
        enhanced_coverage >= float(min_feature_coverage)
        and intraday_rate >= 0.60
        and auction_rate >= 0.60
    )
    contract = FEATURE_CONTRACT if use_enhanced else CORE_FEATURE_CONTRACT
    return contract, {
        "feature_mode": "enhanced" if use_enhanced else "core",
        "core_feature_coverage": core_coverage,
        "enhanced_feature_coverage": enhanced_coverage,
        "intraday_available_rate": intraday_rate,
        "auction_available_rate": auction_rate,
    }


def _resolve_eligible_trade_dates(explicit: Optional[Sequence[str]] = None) -> Optional[set[str]]:
    if explicit:
        out = {str(x).strip() for x in explicit if re.match(r"^\d{8}$", str(x).strip())}
        return out or None

    raw = os.getenv("A_TOP10_ELIGIBLE_TRADE_DATES", "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            out = {str(x).strip() for x in obj if re.match(r"^\d{8}$", str(x).strip())}
            return out or None
    except Exception:
        pass

    out = {x.strip() for x in raw.split(",") if re.match(r"^\d{8}$", x.strip())}
    return out or None


def _build_X_y_from_feature_history(
    s,
    lookback: int,
    eligible_trade_dates: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    hist = _load_feature_history(s)
    hist = _normalize_id_columns(hist)

    required = {"trade_date", "ts_code", "y_limit_hit", "is_sample_mature"}
    if hist.empty or not required.issubset(set(hist.columns)):
        return np.zeros((0, len(CORE_FEATURES))), np.zeros((0,)), {
            "mature_samples": 0,
            "positive_samples": 0,
            "feature_coverage": 0.0,
            "reason": "feature_history_missing_or_unlabeled",
            "eligible_trade_dates_count": 0,
            "features": list(CORE_FEATURE_CONTRACT),
            "sample_trade_dates": [],
        }

    eligible_set = _resolve_eligible_trade_dates(eligible_trade_dates)

    hist["trade_date"] = hist["trade_date"].astype(str).str.strip()
    if eligible_set is not None:
        hist = hist[hist["trade_date"].isin(eligible_set)].copy()

    dates = sorted(hist["trade_date"].unique()) if not hist.empty else []
    if lookback > 0:
        dates = dates[-lookback:]

    selected_hist = hist[hist["trade_date"].isin(dates)].copy()
    feature_names, selection_meta = _select_training_feature_contract(
        selected_hist,
        min_feature_coverage=_get_training_policy(s).min_feature_coverage,
    )

    rows = []
    y = []
    sample_trade_dates: List[str] = []
    coverage_list = []

    for d in dates:
        df_day = hist[hist["trade_date"] == str(d)].copy()
        if df_day.empty:
            continue

        mature = pd.to_numeric(df_day.get("is_sample_mature"), errors="coerce").fillna(0.0)
        label = pd.to_numeric(df_day.get("y_limit_hit"), errors="coerce")

        if "learnable_flag" in df_day.columns:
            learnable = pd.to_numeric(df_day.get("learnable_flag"), errors="coerce").fillna(0.0)
            df_day = df_day[(mature > 0.5) & (learnable > 0.5) & label.notna()].copy()
        else:
            df_day = df_day[(mature > 0.5) & label.notna()].copy()

        if df_day.empty:
            continue

        coverage_list.append(_sample_feature_coverage(df_day, feature_names))

        feat = _ensure_inference_input(df_day)
        yy = pd.to_numeric(df_day["y_limit_hit"], errors="coerce").fillna(0.0).astype(int)

        for i in range(len(df_day)):
            rows.append(feat[list(feature_names)].iloc[i].astype(float).values)
            y.append(int(yy.iloc[i]))
            sample_trade_dates.append(str(d))

    if not rows:
        return np.zeros((0, len(feature_names))), np.zeros((0,)), {
            "mature_samples": 0,
            "positive_samples": 0,
            "feature_coverage": 0.0,
            "reason": "no_mature_labeled_rows",
            "eligible_trade_dates_count": len(eligible_set) if eligible_set is not None else 0,
            "features": list(feature_names),
            "sample_trade_dates": [],
            **selection_meta,
        }

    X = np.asarray(rows, dtype=float)
    yy = np.asarray(y, dtype=int)
    meta = {
        "mature_samples": int(X.shape[0]),
        "positive_samples": int(yy.sum()),
        "feature_coverage": float(np.mean(coverage_list)) if coverage_list else 0.0,
        "reason": "ok",
        "eligible_trade_dates_count": len(eligible_set) if eligible_set is not None else 0,
        "features": list(feature_names),
        "sample_trade_dates": sample_trade_dates,
        **selection_meta,
    }
    return X, yy, meta


def _training_gate_summary(
    n_samples: int,
    pos_samples: int,
    feature_coverage: float,
    policy: Step5TrainingPolicy,
) -> Dict[str, Any]:
    level = _classify_training_level(n_samples, policy)
    formal_update_allowed = _can_formal_update(level, policy)

    summary = {
        "run_mode": policy.run_mode,
        "allow_model_update_raw": policy.allow_model_update_raw,
        "allow_model_update": bool(policy.allow_model_update),
        "level": level,
        "mature_samples": int(n_samples),
        "positive_samples": int(pos_samples),
        "feature_coverage": float(feature_coverage),
        "formal_update_allowed": bool(formal_update_allowed),
        "trial_training_only": level == "level1",
    }

    if level == "below_level1":
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "below_level1_min_samples"})
        return summary

    if pos_samples < policy.min_positive_samples:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "insufficient_positive_samples"})
        return summary

    if feature_coverage < policy.min_feature_coverage:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "insufficient_feature_coverage"})
        return summary

    summary.update({"ok": True, "trained": True, "updated": False, "reason": "ready_for_training"})
    return summary


def _fit_time_calibrated_model(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_trade_dates: Sequence[str],
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Fit on earlier dates and calibrate on a trailing, non-overlapping window."""
    dates = np.asarray([str(x) for x in sample_trade_dates], dtype=object)
    unique_dates = sorted(set(dates.tolist()))
    meta: Dict[str, Any] = {
        "method": "sigmoid",
        "split": "trailing_trade_dates",
        "unique_trade_dates": len(unique_dates),
    }
    if len(dates) != len(y) or len(unique_dates) < 12:
        meta["reason"] = "insufficient_trade_dates_for_calibration"
        return None, meta

    # First run an untouched trailing validation. This is the promotion gate;
    # those labels are not used by the candidate model or its calibrator.
    holdout_days = max(3, int(np.ceil(len(unique_dates) * 0.10)))
    holdout_days = min(holdout_days, len(unique_dates) - 8)
    holdout_start = unique_dates[-holdout_days]
    fit_mask = dates < holdout_start
    holdout_mask = ~fit_mask
    fit_dates = sorted(set(dates[fit_mask].tolist()))
    validation_calibration_days = max(4, int(np.ceil(len(fit_dates) * 0.20)))
    validation_calibration_days = min(validation_calibration_days, len(fit_dates) - 3)
    validation_calibration_start = fit_dates[-validation_calibration_days]
    validation_test_fold = np.where(
        dates[fit_mask] >= validation_calibration_start,
        0,
        -1,
    )
    validation_train_y = y[fit_mask][validation_test_fold < 0]
    validation_calibration_y = y[fit_mask][validation_test_fold == 0]
    holdout_y = y[holdout_mask]
    if (
        len(np.unique(validation_train_y)) < 2
        or len(np.unique(validation_calibration_y)) < 2
        or len(np.unique(holdout_y)) < 2
    ):
        meta["reason"] = "validation_split_single_class"
        return None, meta

    candidate = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=PredefinedSplit(test_fold=validation_test_fold),
        ensemble=True,
    )
    candidate.fit(X.loc[fit_mask], y[fit_mask])
    holdout_prob = np.asarray(candidate.predict_proba(X.loc[holdout_mask]))[:, 1]
    holdout_brier = float(np.mean((holdout_prob - holdout_y) ** 2))
    baseline_prob = float(np.mean(validation_train_y))
    baseline_brier = float(np.mean((baseline_prob - holdout_y) ** 2))
    p_at_1_hits: List[int] = []
    holdout_dates = dates[holdout_mask]
    for day in sorted(set(holdout_dates.tolist())):
        day_pos = np.flatnonzero(holdout_dates == day)
        if len(day_pos):
            p_at_1_hits.append(int(holdout_y[day_pos[int(np.argmax(holdout_prob[day_pos]))]]))
    holdout_p_at_1 = float(np.mean(p_at_1_hits)) if p_at_1_hits else 0.0
    meta["validation"] = {
        "holdout_start": holdout_start,
        "holdout_days": int(holdout_days),
        "holdout_rows": int(holdout_mask.sum()),
        "brier": holdout_brier,
        "constant_base_rate_brier": baseline_brier,
        "p_at_1": holdout_p_at_1,
        "positive_rate": float(np.mean(holdout_y)),
    }
    if holdout_brier > min(0.30, baseline_brier * 1.20):
        meta["reason"] = "validation_brier_gate_failed"
        return None, meta

    calibration_days = max(5, int(np.ceil(len(unique_dates) * 0.20)))
    calibration_days = min(calibration_days, len(unique_dates) - 3)
    calibration_start = unique_dates[-calibration_days]
    is_calibration = dates >= calibration_start
    train_y = y[~is_calibration]
    calibration_y = y[is_calibration]
    if len(np.unique(train_y)) < 2 or len(np.unique(calibration_y)) < 2:
        meta["reason"] = "calibration_split_single_class"
        return None, meta

    test_fold = np.where(is_calibration, 0, -1)
    calibrated = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=PredefinedSplit(test_fold=test_fold),
        ensemble=True,
    )
    calibrated.fit(X, y)
    meta.update(
        {
            "reason": "ok",
            "calibration_start": calibration_start,
            "train_rows": int((~is_calibration).sum()),
            "calibration_rows": int(is_calibration.sum()),
            "calibration_days": int(calibration_days),
        }
    )
    return calibrated, meta


def train_step5_lr(
    s,
    lookback: int = 120,
    theme_file_name: str = "step4_theme.csv",
    eligible_trade_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    保留旧签名兼容，但 V3 正式训练不再依赖 step4_theme.csv，
    而是依赖已成熟、已打标签且已通过 Step7 批门筛选的 feature_history.csv。
    """
    policy = _get_training_policy(s)
    X, y, meta = _build_X_y_from_feature_history(s, lookback=lookback, eligible_trade_dates=eligible_trade_dates)

    n_samples = int(meta.get("mature_samples", 0))
    pos_samples = int(meta.get("positive_samples", 0))
    feature_coverage = float(meta.get("feature_coverage", 0.0))

    summary = _training_gate_summary(
        n_samples=n_samples,
        pos_samples=pos_samples,
        feature_coverage=feature_coverage,
        policy=policy,
    )
    summary["data_reason"] = meta.get("reason", "")
    summary["eligible_trade_dates_count"] = int(meta.get("eligible_trade_dates_count", 0))
    for key in [
        "feature_mode",
        "core_feature_coverage",
        "enhanced_feature_coverage",
        "intraday_available_rate",
        "auction_available_rate",
    ]:
        summary[key] = meta.get(key)
    feature_names = tuple(meta.get("features") or CORE_FEATURE_CONTRACT)
    summary["features"] = list(feature_names)
    summary["feature_schema_version"] = _feature_schema_version(feature_names)

    if not summary.get("ok"):
        return summary

    if len(np.unique(y)) < 2:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "single_class_labels"})
        return summary

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    train_X = pd.DataFrame(X, columns=feature_names)
    model, calibration_meta = _fit_time_calibrated_model(
        model,
        train_X,
        y,
        meta.get("sample_trade_dates") or [],
    )
    summary["calibration"] = calibration_meta
    if model is None:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": calibration_meta.get("reason")})
        return summary
    _attach_model_manifest(
        model,
        "logistic",
        features=feature_names,
        probability_is_calibrated=True,
        calibration=calibration_meta,
    )
    summary["trained"] = True

    level = str(summary.get("level", ""))
    if level == "level1":
        summary["updated"] = False
        summary["reason"] = "level1_trial_training_no_formal_update"
        return summary

    if not summary.get("formal_update_allowed", False):
        summary["updated"] = False
        summary["reason"] = "model_update_not_allowed_by_run_mode"
        return summary

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lr_path)
    summary["updated"] = True
    summary["path"] = str(paths.lr_path)
    summary["reason"] = "formal_model_updated"
    return summary


def train_step5_hgb(
    s,
    lookback: int = 150,
    theme_file_name: str = "step4_theme.csv",
    eligible_trade_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train the production nonlinear core model without optional native libs."""
    policy = _get_training_policy(s)
    X, y, meta = _build_X_y_from_feature_history(
        s,
        lookback=lookback,
        eligible_trade_dates=eligible_trade_dates,
    )
    summary = _training_gate_summary(
        n_samples=int(meta.get("mature_samples", 0)),
        pos_samples=int(meta.get("positive_samples", 0)),
        feature_coverage=float(meta.get("feature_coverage", 0.0)),
        policy=policy,
    )
    summary["data_reason"] = meta.get("reason", "")
    summary["eligible_trade_dates_count"] = int(meta.get("eligible_trade_dates_count", 0))
    for key in [
        "feature_mode",
        "core_feature_coverage",
        "enhanced_feature_coverage",
        "intraday_available_rate",
        "auction_available_rate",
    ]:
        summary[key] = meta.get(key)
    feature_names = tuple(meta.get("features") or CORE_FEATURE_CONTRACT)
    summary["features"] = list(feature_names)
    summary["feature_schema_version"] = _feature_schema_version(feature_names)

    if not summary.get("ok"):
        return summary
    if len(np.unique(y)) < 2:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "single_class_labels"})
        return summary

    estimator = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.10,
        random_state=42,
    )
    train_X = pd.DataFrame(X, columns=feature_names)
    model, calibration_meta = _fit_time_calibrated_model(
        estimator,
        train_X,
        y,
        meta.get("sample_trade_dates") or [],
    )
    summary["calibration"] = calibration_meta
    if model is None:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": calibration_meta.get("reason")})
        return summary

    _attach_model_manifest(
        model,
        "hist_gradient_boosting",
        features=feature_names,
        probability_is_calibrated=True,
        calibration=calibration_meta,
    )
    summary["trained"] = True

    if str(summary.get("level", "")) == "level1":
        summary["updated"] = False
        summary["reason"] = "level1_trial_training_no_formal_update"
        return summary
    if not summary.get("formal_update_allowed", False):
        summary["updated"] = False
        summary["reason"] = "model_update_not_allowed_by_run_mode"
        return summary

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.hgb_path)
    summary["updated"] = True
    summary["path"] = str(paths.hgb_path)
    summary["reason"] = "formal_model_updated"
    return summary


def train_step5_lgbm(
    s,
    lookback: int = 150,
    theme_file_name: str = "step4_theme.csv",
    eligible_trade_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    policy = _get_training_policy(s)

    if LGBMClassifier is None:
        return {
            "ok": False,
            "trained": False,
            "updated": False,
            "run_mode": policy.run_mode,
            "reason": "lightgbm_not_installed",
        }

    X, y, meta = _build_X_y_from_feature_history(s, lookback=lookback, eligible_trade_dates=eligible_trade_dates)
    n_samples = int(meta.get("mature_samples", 0))
    pos_samples = int(meta.get("positive_samples", 0))
    feature_coverage = float(meta.get("feature_coverage", 0.0))

    summary = _training_gate_summary(
        n_samples=n_samples,
        pos_samples=pos_samples,
        feature_coverage=feature_coverage,
        policy=policy,
    )
    summary["data_reason"] = meta.get("reason", "")
    summary["eligible_trade_dates_count"] = int(meta.get("eligible_trade_dates_count", 0))
    for key in [
        "feature_mode",
        "core_feature_coverage",
        "enhanced_feature_coverage",
        "intraday_available_rate",
        "auction_available_rate",
    ]:
        summary[key] = meta.get(key)
    feature_names = tuple(meta.get("features") or CORE_FEATURE_CONTRACT)
    summary["features"] = list(feature_names)
    summary["feature_schema_version"] = _feature_schema_version(feature_names)

    if not summary.get("ok"):
        return summary

    if len(np.unique(y)) < 2:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": "single_class_labels"})
        return summary

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
    )
    train_X = pd.DataFrame(X, columns=feature_names)
    model, calibration_meta = _fit_time_calibrated_model(
        model,
        train_X,
        y,
        meta.get("sample_trade_dates") or [],
    )
    summary["calibration"] = calibration_meta
    if model is None:
        summary.update({"ok": False, "trained": False, "updated": False, "reason": calibration_meta.get("reason")})
        return summary
    _attach_model_manifest(
        model,
        "lightgbm",
        features=feature_names,
        probability_is_calibrated=True,
        calibration=calibration_meta,
    )
    summary["trained"] = True

    level = str(summary.get("level", ""))
    if level == "level1":
        summary["updated"] = False
        summary["reason"] = "level1_trial_training_no_formal_update"
        return summary

    if not summary.get("formal_update_allowed", False):
        summary["updated"] = False
        summary["reason"] = "model_update_not_allowed_by_run_mode"
        return summary

    paths = _get_model_paths(s=s)
    _save_joblib(model, paths.lgbm_path)
    summary["updated"] = True
    summary["path"] = str(paths.lgbm_path)
    summary["reason"] = "formal_model_updated"
    return summary


def train_step5_models(
    s,
    lookback: int = 150,
    theme_file_name: str = "step4_theme.csv",
    eligible_trade_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    res_lr = train_step5_lr(s, lookback=lookback, theme_file_name=theme_file_name, eligible_trade_dates=eligible_trade_dates)
    res_hgb = train_step5_hgb(s, lookback=lookback, theme_file_name=theme_file_name, eligible_trade_dates=eligible_trade_dates)
    res_lgbm = train_step5_lgbm(s, lookback=lookback, theme_file_name=theme_file_name, eligible_trade_dates=eligible_trade_dates)
    return {
        "ok": bool(res_lr.get("ok")) or bool(res_hgb.get("ok")) or bool(res_lgbm.get("ok")),
        "updated": bool(res_lr.get("updated")) or bool(res_hgb.get("updated")) or bool(res_lgbm.get("updated")),
        "run_mode": _resolve_run_mode(),
        "lr": res_lr,
        "hgb": res_hgb,
        "lgbm": res_lgbm,
    }


# =========================================================
# Probability layers
# =========================================================

def _calc_prob_rule(df: pd.DataFrame) -> pd.Series:
    """
    规则概率：
    - 只用于主模型缺失 / 融合补充
    - 内部用数值兜底，不反写上游契约字段
    """
    feat = _ensure_inference_input(df)

    strength_axis = feat["strength_plus_score"] if "strength_plus_score" in feat.columns else feat["StrengthScore"]
    strength = _clip01(strength_axis / 100.0)
    theme = _clip01(feat["ThemeBoost"])  # ThemeBoost 本身就是 0~1

    turnover = feat["turnover_rate"].astype(float)
    turnover_score = pd.Series(
        np.where(turnover <= 0, 0.0, np.exp(-((turnover - 18.0) / 12.0) ** 2)),
        index=df.index,
        dtype="float64",
    )
    turnover_score = _clip01(turnover_score)

    seal_amount = feat["seal_amount"].astype(float)
    seal_score = _clip01(np.log1p(np.maximum(seal_amount, 0.0)) / 16.0)

    open_times = feat["open_times"].astype(float)
    open_score = _clip01(1.0 - np.minimum(np.maximum(open_times, 0.0), 8.0) / 8.0)

    rule = (
        0.42 * strength
        + 0.18 * theme
        + 0.16 * seal_score
        + 0.14 * open_score
        + 0.10 * turnover_score
    )
    return _clip01(rule)


def _predict_with_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    index: pd.Index,
    feature_names: Sequence[str] = FEATURE_CONTRACT,
) -> ModelPredictionResult:
    contract = _validate_model_contract(model, expected_features=feature_names)
    empty = pd.Series([np.nan] * len(index), index=index, dtype="float64")
    if not contract.is_valid:
        return ModelPredictionResult(
            values=empty,
            contract=contract,
            prediction_status="rejected",
        )

    estimator = _unwrap_model(model)
    expected = tuple(str(x) for x in feature_names)
    try:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in expected if c not in X.columns]
            if missing:
                raise ValueError(f"inference input missing ordered features: {missing}")
            model_input: pd.DataFrame | np.ndarray = X.loc[:, list(expected)]
        else:
            model_input = np.asarray(X, dtype=float)
            if model_input.ndim != 2 or model_input.shape[1] != len(expected):
                shape = tuple(model_input.shape)
                raise ValueError(
                    f"inference input shape mismatch: expected_columns={len(expected)}; actual_shape={shape}"
                )

        if hasattr(estimator, "predict_proba"):
            raw = np.asarray(estimator.predict_proba(model_input))
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise ValueError(f"predict_proba must return at least two columns; actual_shape={raw.shape}")
            predicted = raw[:, 1]
        elif hasattr(estimator, "predict"):
            predicted = np.asarray(estimator.predict(model_input))
        else:
            raise TypeError("model exposes neither predict_proba nor predict")

        predicted = np.asarray(predicted, dtype=float).reshape(-1)
        if len(predicted) != len(index):
            raise ValueError(
                f"prediction row count mismatch: expected_rows={len(index)}; actual_rows={len(predicted)}"
            )
        values = _clip01(pd.Series(predicted, index=index, dtype="float64"))
        status = "ok" if values.notna().all() else "ok_with_missing_values"
        return ModelPredictionResult(values=values, contract=contract, prediction_status=status)
    except Exception as exc:
        return ModelPredictionResult(
            values=empty,
            contract=contract,
            prediction_status="error",
            prediction_error=f"{type(exc).__name__}: {exc}",
        )


def _calc_prob_ml(
    df: pd.DataFrame,
    s=None,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, ModelPredictionResult]]:
    feat = _ensure_inference_input(df)

    lr_model = load_lr(s)
    hgb_model = load_hgb(s)
    lgbm_model = load_lgbm(s)

    lr_features = _model_feature_contract(lr_model)
    hgb_features = _model_feature_contract(hgb_model)
    lgbm_features = _model_feature_contract(lgbm_model)
    lr_result = _predict_with_model(
        lr_model,
        feat.loc[:, list(lr_features)].astype(float),
        df.index,
        feature_names=lr_features,
    )
    hgb_result = _predict_with_model(
        hgb_model,
        feat.loc[:, list(hgb_features)].astype(float),
        df.index,
        feature_names=hgb_features,
    )
    lgbm_result = _predict_with_model(
        lgbm_model,
        feat.loc[:, list(lgbm_features)].astype(float),
        df.index,
        feature_names=lgbm_features,
    )
    return lr_result.values, hgb_result.values, lgbm_result.values, {
        "lr": lr_result,
        "hgb": hgb_result,
        "lgbm": lgbm_result,
    }


def _fuse_probabilities(
    prob_rule: pd.Series,
    prob_lr: pd.Series,
    prob_hgb: pd.Series,
    prob_lgbm: pd.Series,
    s=None,
) -> Tuple[pd.Series, pd.Series]:
    cfg = _get_ml_cfg(s)
    model_pref = str(cfg.get("model", "auto")).lower()
    fusion_mode = str(cfg.get("fusion_mode", "ml_first")).lower()
    fallback_to_rule = bool(cfg.get("fallback_to_rule", True))
    w_rule = float(cfg.get("rule_weight", 0.30))
    w_ml = float(cfg.get("ml_weight", 0.70))

    w_sum = max(1e-12, w_rule + w_ml)
    w_rule /= w_sum
    w_ml /= w_sum

    if model_pref == "logistic":
        ml_main = prob_lr
        ml_name = "lr"
    elif model_pref in {"hgb", "hist_gradient_boosting"}:
        ml_main = prob_hgb
        ml_name = "hgb"
    elif model_pref == "lightgbm":
        ml_main = prob_lgbm
        ml_name = "lgbm"
    else:
        hgb_ok = prob_hgb.notna()
        lgbm_ok = prob_lgbm.notna()
        lr_ok = prob_lr.notna()
        ml_main = prob_hgb.where(hgb_ok, prob_lgbm.where(lgbm_ok, prob_lr))
        ml_name = None

    Probability = pd.Series([np.nan] * len(prob_rule), index=prob_rule.index, dtype="float64")
    src = pd.Series(["unresolved"] * len(prob_rule), index=prob_rule.index, dtype="object")

    if model_pref in {"logistic", "hgb", "hist_gradient_boosting", "lightgbm"}:
        ml_ok = ml_main.notna()

        if fusion_mode == "weighted":
            fused = w_rule * prob_rule.fillna(0.0) + w_ml * ml_main.fillna(prob_rule.fillna(0.0))
            Probability[:] = fused
            src[:] = np.where(ml_ok, f"blend:{ml_name}+rule", "fallback_rule")
        else:
            if fallback_to_rule:
                Probability[:] = np.where(ml_ok, ml_main, prob_rule)
                src[:] = np.where(ml_ok, ml_name, "fallback_rule")
            else:
                Probability[:] = ml_main
                src[:] = np.where(ml_ok, ml_name, "ml_missing")

    else:
        hgb_ok = prob_hgb.notna()
        lgbm_ok = prob_lgbm.notna()
        lr_ok = prob_lr.notna()

        if fusion_mode == "weighted":
            ml_any = prob_hgb.where(hgb_ok, prob_lgbm.where(lgbm_ok, prob_lr))
            Probability[:] = w_rule * prob_rule.fillna(0.0) + w_ml * ml_any.fillna(prob_rule.fillna(0.0))
            src[:] = np.where(
                hgb_ok,
                "blend:hgb+rule",
                np.where(lgbm_ok, "blend:lgbm+rule", np.where(lr_ok, "blend:lr+rule", "fallback_rule")),
            )
        else:
            if fallback_to_rule:
                Probability[:] = np.where(
                    hgb_ok,
                    prob_hgb,
                    np.where(lgbm_ok, prob_lgbm, np.where(lr_ok, prob_lr, prob_rule)),
                )
                src[:] = np.where(
                    hgb_ok,
                    "hgb",
                    np.where(lgbm_ok, "lgbm", np.where(lr_ok, "lr", "fallback_rule")),
                )
            else:
                Probability[:] = np.where(hgb_ok, prob_hgb, np.where(lgbm_ok, prob_lgbm, prob_lr))
                src[:] = np.where(hgb_ok, "hgb", np.where(lgbm_ok, "lgbm", np.where(lr_ok, "lr", "ml_missing")))

    Probability = _clip01(Probability.fillna(prob_rule.fillna(0.0)))
    return Probability.astype("float64"), src.astype("object")


def _not_run_model_result(index: pd.Index, status: str, reason: str) -> ModelPredictionResult:
    return ModelPredictionResult(
        values=pd.Series([np.nan] * len(index), index=index, dtype="float64"),
        contract=_empty_model_contract(status, reason),
        prediction_status="not_run",
    )


def _model_result_reason(result: ModelPredictionResult) -> str:
    parts = [f"contract={result.contract.status}", result.contract.reason]
    parts.append(f"prediction={result.prediction_status}")
    if result.prediction_error:
        parts.append(f"error={result.prediction_error}")
    return "; ".join(x for x in parts if x)


def _model_audit_summary(results: Dict[str, ModelPredictionResult]) -> str:
    return " | ".join(
        f"{name}[{_model_result_reason(result)}]"
        for name, result in results.items()
    )


def _source_model_name(source: str) -> Optional[str]:
    source = str(source).lower()
    if "hgb" in source:
        return "hgb"
    if "lgbm" in source:
        return "lgbm"
    if "lr" in source:
        return "lr"
    return None


def _add_probability_audit_columns(
    out: pd.DataFrame,
    probability: pd.Series,
    sources: pd.Series,
    model_results: Dict[str, ModelPredictionResult],
) -> pd.DataFrame:
    out = out.copy()

    for name in ["lr", "hgb", "lgbm"]:
        result = model_results[name]
        out[f"{name}_model_contract_status"] = result.contract.status
        out[f"{name}_model_contract_reason"] = result.contract.reason
        out[f"{name}_model_prediction_status"] = result.prediction_status
        out[f"{name}_model_prediction_error"] = result.prediction_error

    fallback_reason = f"rule fallback is an uncalibrated rank score; {_model_audit_summary(model_results)}"
    statuses: List[str] = []
    reasons: List[str] = []
    calibrated_flags: List[bool] = []
    semantics: List[str] = []
    schema_versions: List[str] = []

    for source in sources.astype(str):
        model_name = _source_model_name(source)
        result = model_results.get(model_name) if model_name else None

        if result is None:
            statuses.append("fallback_uncalibrated" if source == "fallback_rule" else "no_valid_model_output")
            reasons.append(fallback_reason)
            calibrated = False
            schema_versions.append("")
        else:
            statuses.append(result.contract.status)
            reason = _model_result_reason(result)
            if source.startswith("blend:"):
                reason = f"{reason}; rule blend is uncalibrated"
            reasons.append(reason)
            calibrated = bool(
                source in {"lr", "hgb", "lgbm"}
                and result.contract.is_valid
                and result.prediction_status == "ok"
                and result.contract.probability_is_calibrated
            )
            schema_versions.append(str(result.contract.feature_schema_version or ""))

        calibrated_flags.append(calibrated)
        semantics.append("calibrated_probability" if calibrated else "rank_score_uncalibrated")

    out["rank_score"] = pd.to_numeric(probability, errors="coerce").astype("float64")
    out["probability_is_calibrated"] = pd.Series(calibrated_flags, index=out.index, dtype="bool")
    out["p_limit_up_calibrated"] = out["rank_score"].where(out["probability_is_calibrated"], np.nan)
    out["probability_semantics"] = pd.Series(semantics, index=out.index, dtype="object")
    out["model_contract_status"] = pd.Series(statuses, index=out.index, dtype="object")
    out["model_contract_reason"] = pd.Series(reasons, index=out.index, dtype="object")
    out["model_schema_version"] = pd.Series(schema_versions, index=out.index, dtype="object")
    return out


# =========================================================
# Main inference
# =========================================================

def run_step5(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    raw_input = _ensure_df(theme_df)
    if raw_input.empty:
        out = pd.DataFrame()
        for c in ["prob_lr", "prob_hgb", "prob_lgbm", "prob_rule", "Probability", "rank_score", "p_limit_up_calibrated"]:
            out[c] = pd.Series(dtype="float64")
        out["probability_is_calibrated"] = pd.Series(dtype="bool")
        for c in [
            "probability_semantics",
            "model_contract_status",
            "model_contract_reason",
            "model_schema_version",
            "lr_model_contract_status",
            "lr_model_contract_reason",
            "lr_model_prediction_status",
            "lr_model_prediction_error",
            "hgb_model_contract_status",
            "hgb_model_contract_reason",
            "hgb_model_prediction_status",
            "hgb_model_prediction_error",
            "lgbm_model_contract_status",
            "lgbm_model_contract_reason",
            "lgbm_model_prediction_status",
            "lgbm_model_prediction_error",
            "_prob_src",
        ]:
            out[c] = pd.Series(dtype="object")
        return out

    trade_date = _guess_trade_date(raw_input)
    outputs_dir = _get_outputs_dir(s)

    out = _normalize_id_columns(raw_input)
    out = _backfill_features_from_step3(out, trade_date=trade_date, outputs_dir=outputs_dir)

    cfg = _get_ml_cfg(s)
    enable_rule = bool(cfg.get("enable_rule", True))
    enable_ml = bool(cfg.get("enable_ml", True))

    if enable_rule:
        out["prob_rule"] = _calc_prob_rule(out)
    else:
        out["prob_rule"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")

    if enable_ml:
        prob_lr, prob_hgb, prob_lgbm, model_results = _calc_prob_ml(out, s=s)
        out["prob_lr"] = prob_lr
        out["prob_hgb"] = prob_hgb
        out["prob_lgbm"] = prob_lgbm
    else:
        out["prob_lr"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
        out["prob_hgb"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
        out["prob_lgbm"] = pd.Series([np.nan] * len(out), index=out.index, dtype="float64")
        model_results = {
            "lr": _not_run_model_result(out.index, "ml_disabled", "ML inference is disabled by configuration"),
            "hgb": _not_run_model_result(out.index, "ml_disabled", "ML inference is disabled by configuration"),
            "lgbm": _not_run_model_result(out.index, "ml_disabled", "ML inference is disabled by configuration"),
        }

    Probability, _prob_src = _fuse_probabilities(
        prob_rule=pd.to_numeric(out["prob_rule"], errors="coerce"),
        prob_lr=pd.to_numeric(out["prob_lr"], errors="coerce"),
        prob_hgb=pd.to_numeric(out["prob_hgb"], errors="coerce"),
        prob_lgbm=pd.to_numeric(out["prob_lgbm"], errors="coerce"),
        s=s,
    )

    out["Probability"] = Probability.astype("float64")
    out["_prob_src"] = _prob_src.astype("object")
    out = _add_probability_audit_columns(
        out,
        probability=out["Probability"],
        sources=out["_prob_src"],
        model_results=model_results,
    )
    out["run_mode"] = _resolve_run_mode()
    out["run_time_utc"] = _utc_now_iso()

    try:
        debug_path = outputs_dir / "learning" / "step5_debug_latest.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug = {
            "trade_date": trade_date,
            "intraday_features_in_ml": list(ML_INTRADAY_FEATURES),
            "feature_count": int(len(FEATURES)),
            "intraday_feature_coverage": {
                "intraday_available_ratio": float(pd.to_numeric(out.get("intraday_available"), errors="coerce").fillna(0).gt(0).mean()) if len(out) else 0.0,
                "auction_available_ratio": float(pd.to_numeric(out.get("auction_available"), errors="coerce").fillna(0).gt(0).mean()) if len(out) else 0.0,
            },
            "prob_source_counts": out["_prob_src"].value_counts(dropna=False).to_dict() if "_prob_src" in out.columns else {},
            "probability_semantics_counts": out["probability_semantics"].value_counts(dropna=False).to_dict(),
            "model_contracts": {
                name: {
                    "status": result.contract.status,
                    "reason": result.contract.reason,
                    "expected_feature_count": result.contract.expected_feature_count,
                    "actual_feature_count": result.contract.actual_feature_count,
                    "manifest_schema_version": result.contract.manifest_schema_version,
                    "feature_schema_version": result.contract.feature_schema_version,
                    "probability_is_calibrated": result.contract.probability_is_calibrated,
                    "prediction_status": result.prediction_status,
                    "prediction_error": result.prediction_error,
                }
                for name, result in model_results.items()
            },
        }
        debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    clip_min = float(cfg.get("clip_min", 0.0))
    clip_max = float(cfg.get("clip_max", 1.0))
    for c in ["prob_lr", "prob_hgb", "prob_lgbm", "prob_rule", "Probability", "rank_score", "p_limit_up_calibrated"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").clip(clip_min, clip_max)

    sort_cols = ["Probability"]
    if "StrengthScore" in out.columns:
        sort_cols.append("StrengthScore")
    ascending = [False] * len(sort_cols)
    out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    prefer_order = [
        "trade_date",
        "ts_code",
        "name",
        "StrengthScore",
        "ThemeBoost",
        "seal_amount",
        "open_times",
        "turnover_rate",
        "prob_lr",
        "prob_hgb",
        "prob_lgbm",
        "prob_rule",
        "Probability",
        "p_limit_up_calibrated",
        "rank_score",
        "probability_is_calibrated",
        "probability_semantics",
        "model_contract_status",
        "model_contract_reason",
        "model_schema_version",
        "lr_model_contract_status",
        "lr_model_contract_reason",
        "lr_model_prediction_status",
        "lr_model_prediction_error",
        "hgb_model_contract_status",
        "hgb_model_contract_reason",
        "hgb_model_prediction_status",
        "hgb_model_prediction_error",
        "lgbm_model_contract_status",
        "lgbm_model_contract_reason",
        "lgbm_model_prediction_status",
        "lgbm_model_prediction_error",
        "_prob_src",
        "run_mode",
        "run_time_utc",
    ]
    exist = [c for c in prefer_order if c in out.columns]
    others = [c for c in out.columns if c not in exist]
    out = out[exist + others]

    return out


def run(theme_df: pd.DataFrame, s=None) -> pd.DataFrame:
    return run_step5(theme_df=theme_df, s=s)
