import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Feature_Engineering import prepare_training_data, DB_PATH

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_FULL_PATH = os.path.join(DATA_DIR, 'twstock.db')

STOCK_ID = '2497'  # Default stock for testing (TSMC)
TARGET_DAYS = 3

def create_labels(df):
    """
    Creates labels for 2-Stage Classification.
    """
    # --- Stage 1: Direction (3 Classes) ---
    # 0: Down (Return < -1%)
    # 1: Neutral (-1% <= Return <= 1%)
    # 2: Up (Return > 1%)
    conditions_s1 = [
        (df['Future_Return'] < -0.01),
        (df['Future_Return'] >= -0.01) & (df['Future_Return'] <= 0.01),
        (df['Future_Return'] > 0.01)
    ]
    choices_s1 = [0, 1, 2]
    df['Label_Stage1'] = np.select(conditions_s1, choices_s1, default=1)

    # --- Stage 2 Up: Magnitude (2 Classes) ---
    # Only relevant if Stage 1 is Up (2)
    # 0: Moderate Up (1% < Return <= 5%)
    # 1: Strong Up (Return > 5%)
    conditions_s2_up = [
        (df['Future_Return'] > 0.01) & (df['Future_Return'] <= 0.05),
        (df['Future_Return'] > 0.05)
    ]
    choices_s2_up = [0, 1]
    # Default to -1 for non-up rows (will be filtered out during training)
    df['Label_Stage2_Up'] = np.select(conditions_s2_up, choices_s2_up, default=-1)

    # --- Stage 2 Down: Magnitude (2 Classes) ---
    # Only relevant if Stage 1 is Down (0)
    # 0: Moderate Down (-5% <= Return < -1%)
    # 1: Strong Down (Return < -5%)
    conditions_s2_down = [
        (df['Future_Return'] >= -0.05) & (df['Future_Return'] < -0.01),
        (df['Future_Return'] < -0.05)
    ]
    choices_s2_down = [0, 1]
    # Default to -1 for non-down rows
    df['Label_Stage2_Down'] = np.select(conditions_s2_down, choices_s2_down, default=-1)

    return df

def train_xgb_classifier(X, y, model_name, num_class):
    """
    Helper function to train a single XGBoost classifier.
    """
    print(f"\nTraining {model_name}...")
    
    # Split Data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Check classes
    unique_classes = np.unique(y_train)
    print(f"Classes in training set: {unique_classes}")
    
    if len(unique_classes) < 2:
        print(f"Warning: Only 1 class found in training data for {model_name}. Skipping training.")
        return None

    # Determine Objective and Eval Metric based on num_class
    if num_class == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'

    # Initialize Model
    model = xgb.XGBClassifier(
        objective=objective,
        num_class=num_class if num_class > 2 else None, # binary doesn't need num_class
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric=eval_metric
    )
    
    # Train
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
    except Exception as e:
        print(f"Error with early stopping: {e}. Retrying without it.")
        model.fit(X_train, y_train, verbose=False)
        
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    
    # Generate target names for report
    if num_class == 3: # Stage 1
        target_names = ['Down (0)', 'Neutral (1)', 'Up (2)']
    elif num_class == 2: # Stage 2
        target_names = ['Moderate (0)', 'Strong (1)']
    else:
        target_names = None
        
    # Handle missing classes in test set for report
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    if target_names:
        present_names = [target_names[i] for i in unique_labels if i < len(target_names)]
    else:
        present_names = None

    print(classification_report(y_test, y_pred, target_names=present_names, labels=unique_labels, zero_division=0))
    
    return model

def train_2stage_models(stock_id=STOCK_ID, save_model=True):
    print(f"--- Starting 2-Stage Model Training for {stock_id} ---")
    
    # 1. Load Data
    df = prepare_training_data(stock_id, target_days=TARGET_DAYS, path=DB_FULL_PATH)
    if df is None or df.empty:
        print("Error: No data found.")
        return

    # 2. Create Labels
    df = create_labels(df)
    
    # 3. Prepare Features (X)
    drop_cols = ['Stock_ID', 'Type', 'Date', 'Future_Return', 
                 'Label_Stage1', 'Label_Stage2_Up', 'Label_Stage2_Down']
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X_all = X_all.select_dtypes(include=[np.number])
    
    # --- Train Stage 1: Direction ---
    y_s1 = df['Label_Stage1']
    model_s1 = train_xgb_classifier(X_all, y_s1, "Stage1_Direction", num_class=3)
    
    # --- Train Stage 2: Up Magnitude ---
    # Filter data where Stage 1 is Up (Label 2)
    mask_up = df['Label_Stage1'] == 2
    if mask_up.sum() > 50: # Ensure enough samples
        X_up = X_all[mask_up]
        y_up = df.loc[mask_up, 'Label_Stage2_Up']
        model_s2_up = train_xgb_classifier(X_up, y_up, "Stage2_Up_Magnitude", num_class=2)
    else:
        print("Not enough 'Up' samples to train Stage 2 Up model.")
        model_s2_up = None

    # --- Train Stage 2: Down Magnitude ---
    # Filter data where Stage 1 is Down (Label 0)
    mask_down = df['Label_Stage1'] == 0
    if mask_down.sum() > 50:
        X_down = X_all[mask_down]
        y_down = df.loc[mask_down, 'Label_Stage2_Down']
        model_s2_down = train_xgb_classifier(X_down, y_down, "Stage2_Down_Magnitude", num_class=2)
    else:
        print("Not enough 'Down' samples to train Stage 2 Down model.")
        model_s2_down = None
        
    # 4. Save Models
    if save_model:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        if model_s1:
            joblib.dump(model_s1, os.path.join(MODELS_DIR, f'xgb_stage1_{stock_id}.pkl'))
        if model_s2_up:
            joblib.dump(model_s2_up, os.path.join(MODELS_DIR, f'xgb_stage2_up_{stock_id}.pkl'))
        if model_s2_down:
            joblib.dump(model_s2_down, os.path.join(MODELS_DIR, f'xgb_stage2_down_{stock_id}.pkl'))
            
        print("\nModels saved successfully.")

if __name__ == "__main__":
    train_2stage_models()
