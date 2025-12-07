# ============================================================
# ENHANCED CATBOOST VERSION (Single Model, GPU-Friendly)
# ------------------------------------------------------------
# Goal:
#   - Much faster than 3-model stacking
#   - Enhanced feature engineering based on best practices
#   - No aggressive post-processing
#   - Reasonable MAE，避免再出现 700+ 这种灾难
#
# Enhanced Features:
#   - Detailed time features (year, month, day, season)
#   - Missing value indicators
#   - Outlier detection flags
#   - Brand-model combinations
#   - Frequency encoding for categorical features
#   - Statistical features (brand/model level stats within CV folds)
#   - Power-displacement ratio and other interaction features
#
# Files required (same folder as this script):
#   - used_car_train_20200313.csv
#   - used_car_testB_20200421.csv
#
# Run:
#   python train_fast_catboost_gpu.py
#
# Output:
#   price_prediction_fast_catboost.csv
# ============================================================

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor, Pool

# ------------------------------------------------------------
# 1. Paths & basic info
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, "used_car_train_20200313.csv")
test_file  = os.path.join(base_dir, "used_car_testB_20200421.csv")

print("\n=======================================================")
print("🚗 USED CAR PRICE — ENHANCED CATBOOST")
print("=======================================================")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 默认是空格分隔
train = pd.read_csv(train_file, sep=" ")
test  = pd.read_csv(test_file, sep=" ")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train columns: {list(train.columns[:8])} ...\n")

# ------------------------------------------------------------
# 2. Enhanced preprocessing / feature engineering
#    —— 整合高级特征工程，提升模型性能
# ------------------------------------------------------------
def preprocess(df: pd.DataFrame, is_train: bool = True, km_clip: tuple = None, 
               outlier_clips: dict = None) -> tuple:
    df = df.copy()

    # 2.1 日期解析 → 详细时间特征
    for col in ["regDate", "creatDate"]:
        df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
        # 处理无效日期
        df.loc[df[col].isnull(), col] = pd.to_datetime('20160101', format='%Y%m%d')

    # 车辆年龄（年）
    df["car_age"] = (df["creatDate"].dt.year - df["regDate"].dt.year)
    df["car_age"] = df["car_age"].clip(lower=0, upper=30)
    
    # 车辆年龄（天数）
    used_days = (df["creatDate"] - df["regDate"]).dt.days
    df["used_days"] = used_days.clip(lower=0, upper=365 * 30)
    df["vehicle_age_years"] = df["used_days"] / 365.0

    # 注册日期特征
    df["reg_year"] = df["regDate"].dt.year
    df["reg_month"] = df["regDate"].dt.month
    df["reg_day"] = df["regDate"].dt.day
    df["reg_season"] = ((df["reg_month"] % 12 + 3) // 3).astype(int)
    
    # 创建日期特征
    df["creat_year"] = df["creatDate"].dt.year
    df["creat_month"] = df["creatDate"].dt.month
    df["creat_day"] = df["creatDate"].dt.day
    df["creat_season"] = ((df["creat_month"] % 12 + 3) // 3).astype(int)
    
    # 是否为新车
    df["is_new_car"] = (df["vehicle_age_years"] < 1).astype(int)
    
    # 相对当前年份的车龄
    current_year = datetime.now().year
    df["car_age_from_now"] = current_year - df["reg_year"]

    # 2.2 缺失值处理 - 所有数值特征
    numerical_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 
                          'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
    for feature in numerical_features:
        if feature in df.columns:
            # 标记缺失值
            if df[feature].isnull().any():
                df[f'{feature}_missing'] = df[feature].isnull().astype(int)
                # 填充缺失值（使用中位数）
                df[feature] = df[feature].fillna(df[feature].median())

    # 2.3 power / kilometer 基础剪裁（非常关键，防止极端值干扰）
    if "power" in df.columns:
        df["power"] = df["power"].clip(20, 600)

    # 用训练集分位数做剪裁，测试集使用训练集的分位数
    if is_train:
        km_low  = df["kilometer"].quantile(0.001)
        km_high = df["kilometer"].quantile(0.999)
        km_clip = (km_low, km_high)
    else:
        if km_clip is not None:
            km_low, km_high = km_clip
        else:
            km_low  = df["kilometer"].quantile(0.001)
            km_high = df["kilometer"].quantile(0.999)
    if "kilometer" in df.columns:
        df["kilometer"] = df["kilometer"].clip(km_low, km_high)

    # 2.3 notRepairedDamage：'-' → NaN → {-1,0,1}
    if df["notRepairedDamage"].dtype == 'object':
        df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
        df["notRepairedDamage"] = df["notRepairedDamage"].map({"0.0": 0, "1.0": 1, 0: 0, 1: 1, "0": 0, "1": 1})
    else:
        df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
    df["notRepairedDamage"] = df["notRepairedDamage"].fillna(-1).astype(int)

    # 2.4 v_0 ~ v_14 统计特征
    v_cols = [c for c in df.columns if c.startswith("v_")]
    if v_cols:
        # 统计特征
        df["v_mean"] = df[v_cols].mean(axis=1)
        df["v_std"]  = df[v_cols].std(axis=1)
        df["v_min"]  = df[v_cols].min(axis=1)
        df["v_max"]  = df[v_cols].max(axis=1)
        df["v_median"] = df[v_cols].median(axis=1)
        
        # 功率与排量比（如果v_0存在）
        if "v_0" in df.columns:
            df["power_displacement_ratio"] = df["power"] / (df["v_0"] + 1)

    # 2.5 异常值处理（基于IQR方法）- 实际裁剪值，不仅仅是标志
    # 参考 feature_engineering_and_catboost.py 的实现
    numerical_cols_for_outlier = ['power', 'kilometer', 'v_0']
    outlier_clips_dict = {}
    
    for col in numerical_cols_for_outlier:
        if col in df.columns:
            if is_train:
                # 训练集：计算IQR并裁剪
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_clips_dict[col] = (lower_bound, upper_bound)
                
                # 创建异常值标志
                df[f'{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                # 实际裁剪异常值
                df[col] = df[col].clip(lower_bound, upper_bound)
            else:
                # 测试集：使用训练集的裁剪范围
                if outlier_clips is not None and col in outlier_clips:
                    lower_bound, upper_bound = outlier_clips[col]
                    # 创建异常值标志（基于训练集的边界）
                    df[f'{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                    # 使用训练集的边界裁剪
                    df[col] = df[col].clip(lower_bound, upper_bound)
                else:
                    # Fallback: 使用当前数据的统计量
                    Q1 = df[col].quantile(0.05)
                    Q3 = df[col].quantile(0.95)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[f'{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                    df[col] = df[col].clip(lower_bound, upper_bound)

    # 2.6 车辆特征组合
    # model转换为数值型（用于特征组合）
    if "model" in df.columns:
        df["model_num"] = df["model"].astype('category').cat.codes
    
    # 品牌与车型组合
    if "brand" in df.columns and "model" in df.columns:
        df["brand_model"] = df["brand"].astype(str) + "_" + df["model"].astype(str)
    
    # 特征组合
    if "power" in df.columns and "model_num" in df.columns:
        df["power_model"] = df["power"] + df["model_num"]

    # 2.7 衍生特征
    df["km_per_year"] = df["kilometer"] / (df["vehicle_age_years"] + 0.1)
    df["power_per_year"] = df["power"] / (df["vehicle_age_years"] + 0.1)
    df["km_x_age"] = df["kilometer"] * df["car_age"]
    df["power_x_age"] = df["power"] * df["car_age"]

    # 2.8 丢掉不直接用于建模的列
    drop_cols = ["regDate", "creatDate"]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df, km_clip, outlier_clips_dict

# 先把 price 留出来
y = train["price"].copy()
train_proc, km_clip, outlier_clips = preprocess(train, is_train=True)
test_proc, _, _ = preprocess(test, is_train=False, km_clip=km_clip, outlier_clips=outlier_clips)

# 目标列 price 仅在 train
train_proc = train_proc.drop(columns=["price"])
# 确保 test 没有 price
test_proc  = test_proc.drop(columns=["price"], errors="ignore")

# 对齐列（交集），防止列不一致
common_cols = sorted(list(set(train_proc.columns) & set(test_proc.columns)))
train_proc = train_proc[common_cols]
test_proc  = test_proc[common_cols]

# 2.9 频率编码（在合并前对训练集和测试集分别处理，避免数据泄漏）
print("创建频率编码特征...")
categorical_cols_for_freq = ["model", "brand", "bodyType", "fuelType", "gearbox", "notRepairedDamage"]
for col in categorical_cols_for_freq:
    if col in train_proc.columns:
        # 使用训练集的频率来编码
        freq_encoding = train_proc[col].value_counts() / len(train_proc)
        train_proc[f'{col}_freq'] = train_proc[col].map(freq_encoding).fillna(0)
        test_proc[f'{col}_freq'] = test_proc[col].map(freq_encoding).fillna(0)

# 更新 common_cols 以包含新特征
common_cols = sorted(list(set(train_proc.columns) & set(test_proc.columns)))
train_proc = train_proc[common_cols]
test_proc  = test_proc[common_cols]

# SaleID 不作为特征
if "SaleID" in common_cols:
    common_cols.remove("SaleID")
    X_train = train_proc[common_cols].copy()
    X_test  = test_proc[common_cols].copy()
    test_saleid = test_proc["SaleID"].values
else:
    X_train = train_proc[common_cols].copy()
    X_test  = test_proc[common_cols].copy()
    test_saleid = test["SaleID"].values

print(f"使用特征数: {X_train.shape[1]}")
print(f"示例特征: {common_cols[:10]} ...\n")

# ------------------------------------------------------------
# 3. Add statistical features (within CV folds to avoid leakage)
# ------------------------------------------------------------
def add_statistical_features(X_tr, y_tr, X_val, X_test=None):
    """
    Add brand-level and model-level statistical features
    Computed only on training fold to avoid data leakage
    """
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    if X_test is not None:
        X_test = X_test.copy()
    
    # Brand-level statistics
    if "brand" in X_tr.columns:
        brand_df = pd.DataFrame({"brand": X_tr["brand"], "price": y_tr.values})
        brand_stats = brand_df.groupby("brand").agg({
            "price": ["mean", "median", "std", "count"]
        })
        brand_stats.columns = ["brand_price_mean", "brand_price_median", "brand_price_std", "brand_count"]
        brand_stats = brand_stats.reset_index()
        
        X_tr = X_tr.merge(brand_stats, on="brand", how="left")
        X_val = X_val.merge(brand_stats, on="brand", how="left")
        if X_test is not None:
            X_test = X_test.merge(brand_stats, on="brand", how="left")
        
        # Fill missing values
        for col in ["brand_count", "brand_price_mean", "brand_price_median", "brand_price_std"]:
            if col in X_tr.columns:
                fill_val = X_tr[col].median() if X_tr[col].dtype in ['float64', 'int64'] else 0
                X_tr[col] = X_tr[col].fillna(fill_val)
                X_val[col] = X_val[col].fillna(fill_val)
                if X_test is not None:
                    X_test[col] = X_test[col].fillna(fill_val)
    
    # Model-level statistics
    if "model" in X_tr.columns:
        model_df = pd.DataFrame({"model": X_tr["model"], "price": y_tr.values})
        model_stats = model_df.groupby("model").agg({
            "price": ["mean", "median", "std", "count"]
        })
        model_stats.columns = ["model_price_mean", "model_price_median", "model_price_std", "model_count"]
        model_stats = model_stats.reset_index()
        
        X_tr = X_tr.merge(model_stats, on="model", how="left")
        X_val = X_val.merge(model_stats, on="model", how="left")
        if X_test is not None:
            X_test = X_test.merge(model_stats, on="model", how="left")
        
        # Fill missing values
        for col in ["model_count", "model_price_mean", "model_price_median", "model_price_std"]:
            if col in X_tr.columns:
                fill_val = X_tr[col].median() if X_tr[col].dtype in ['float64', 'int64'] else 0
                X_tr[col] = X_tr[col].fillna(fill_val)
                X_val[col] = X_val[col].fillna(fill_val)
                if X_test is not None:
                    X_test[col] = X_test[col].fillna(fill_val)
    
    if X_test is not None:
        return X_tr, X_val, X_test
    else:
        return X_tr, X_val

# ------------------------------------------------------------
# 4. CatBoost — 单模型，GPU 优先，5 折 CV
# ------------------------------------------------------------
# 指定哪些列是类别特征
cat_cols = ["model", "brand", "bodyType", "fuelType",
            "gearbox", "regionCode", "seller", "offerType", "name", "brand_model"]
cat_cols = [c for c in cat_cols if c in X_train.columns]

# CatBoost 需要把类别列转成字符串
for c in cat_cols:
    if c in X_train.columns:
        X_train[c] = X_train[c].astype(str)
    if c in X_test.columns:
        X_test[c]  = X_test[c].astype(str)

# Get cat_indices for initial GPU test
cat_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

print(f"类别特征列: {cat_cols}\n")

# log1p 变换目标，预测后再 expm1 回来
y_log = np.log1p(y)

# CatBoost 参数 —— 相比你之前的 8 折 3 模型，这里非常轻量
cat_params = dict(
    loss_function="MAE",
    eval_metric="MAE",
    depth=7,
    learning_rate=0.03,
    iterations=2000,       # 单模型 + GPU，2000 轮很快
    l2_leaf_reg=4.0,
    random_seed=42,
    verbose=200            # 每 200 轮打一行
)

# 尝试用 GPU，失败就自动退回 CPU
try:
    cat_params["task_type"] = "GPU"
    print("尝试使用 GPU 训练 CatBoost ...")
    _tmp_model = CatBoostRegressor(**cat_params)
    # 用一小块数据试跑一下，确认 GPU 可用
    tmp_pool = Pool(X_train.head(200), y_log.head(200), cat_features=cat_indices)
    _tmp_model.fit(tmp_pool)
    print("✅ GPU 模式可用")
except Exception as e:
    print(f"⚠️ GPU 不可用，回退到 CPU. 原因: {str(e)[:80]} ...")
    cat_params["task_type"] = "CPU"

# 正式 5 折 CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_log = np.zeros(len(X_train))
fold_maes = []

start_train = time.time()
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\n📊 Fold {fold}/5 ...")
    X_tr, X_val = X_train.iloc[tr_idx].copy(), X_train.iloc[val_idx].copy()
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
    
    # Add statistical features within fold (avoid data leakage)
    X_tr, X_val = add_statistical_features(X_tr, y_tr, X_val)
    
    # Update cat_indices after adding new features
    cat_cols_current = [c for c in cat_cols if c in X_tr.columns]
    cat_indices_fold = [X_tr.columns.get_loc(c) for c in cat_cols_current if c in X_tr.columns]
    
    # Ensure categorical columns are strings
    for c in cat_cols_current:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(str)
        if c in X_val.columns:
            X_val[c] = X_val[c].astype(str)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_indices_fold)
    val_pool   = Pool(X_val, y_val, cat_features=cat_indices_fold)

    model = CatBoostRegressor(**cat_params)
    # 这里用 early_stopping_rounds=200，防止过拟合，同时也加快训练
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=200
    )

    oof_log[val_idx] = model.predict(val_pool)
    fold_mae = mean_absolute_error(y.iloc[val_idx], np.expm1(oof_log[val_idx]))
    fold_maes.append(fold_mae)
    print(f"   Fold {fold} MAE: {fold_mae:.4f}")

total_oof_mae = mean_absolute_error(y, np.expm1(oof_log))
print("\n================ OOF 结果 ================")
print(f"各折 MAE: {[round(m, 4) for m in fold_maes]}")
print(f"整体 OOF MAE: {total_oof_mae:.4f}")
print("（注意：这是训练集上的交叉验证 MAE，用于大致评估模型，不等于线上分数）")
print("=========================================\n")
print(f"训练耗时: {(time.time() - start_train)/60:.1f} 分钟（你的 RTX 5070 上会更快）")

# ------------------------------------------------------------
# 5. 用全部训练数据，再训一个最终模型，然后预测 testB
# ------------------------------------------------------------
print("\n🚀 使用全部训练数据拟合最终模型，并预测测试集 ...")

# Add statistical features on full training set
X_train_final, _, X_test_final = add_statistical_features(X_train.copy(), y_log, X_test.copy(), X_test.copy())

# Update cat_indices for final model
cat_cols_final = [c for c in cat_cols if c in X_train_final.columns]
cat_indices_final = [X_train_final.columns.get_loc(c) for c in cat_cols_final if c in X_train_final.columns]

# Ensure categorical columns are strings
for c in cat_cols_final:
    if c in X_train_final.columns:
        X_train_final[c] = X_train_final[c].astype(str)
    if c in X_test_final.columns:
        X_test_final[c] = X_test_final[c].astype(str)

full_pool = Pool(X_train_final, y_log, cat_features=cat_indices_final)
final_model = CatBoostRegressor(**cat_params)
final_model.fit(full_pool)

test_pool = Pool(X_test_final, cat_features=cat_indices_final)
pred_log_test = final_model.predict(test_pool)
pred_test = np.expm1(pred_log_test)

# 简单安全剪裁（不要像之前那样瞎放大 / 收缩）
pred_test = np.clip(pred_test, 200, 300000)

# ------------------------------------------------------------
# 5. 保存提交 & 简单分布检查
# ------------------------------------------------------------
sub = pd.DataFrame({
    "SaleID": test_saleid,
    "price": pred_test
})

print("\n📈 Prediction distribution (testB 上的预测分布):")
print(sub["price"].describe([0.01, 0.05, 0.95, 0.99]))

print("\n📊 基本统计:")
print(f"  Mean: {sub['price'].mean():.2f}")
print(f"  Std : {sub['price'].std():.2f}")
print(f"  Min : {sub['price'].min():.2f}")
print(f"  Max : {sub['price'].max():.2f}")

out_file = os.path.join(base_dir, "price_prediction_fast_catboost.csv")
sub.to_csv(out_file, index=False, encoding="utf-8-sig")

print(f"\n💾 已保存提交文件: {out_file}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 ENHANCED CATBOOST VERSION 完成 (with advanced feature engineering)")
