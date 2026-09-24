import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def predict_recurring_merchant(clean_df_path):
    df = pd.read_csv(clean_df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    grouped = df.groupby('client_id')
    
    cutoff = datetime(2026, 1, 1, tzinfo=df['timestamp'].dt.tz)
    cutoff_plus_90 = cutoff + timedelta(days=90)
    preds = {}
    
    for client_id, group in grouped:
        by_cat = group.groupby('candidate_category')
        next_dates = {}
        
        for cat, c_group in by_cat:
            c_group = c_group.sort_values('timestamp')
            
            if len(c_group) < 2: continue
                
            intervals = c_group['timestamp'].diff().dt.days.dropna()
            if len(intervals) == 0: continue
                
            med_interval = intervals.median()
            if med_interval < 5 or med_interval > 380: continue
                
            last_dt = c_group['timestamp'].iloc[-1]
            days_since_last = (cutoff - last_dt).days
            
            if days_since_last > med_interval * 1.8: continue
                
            next_dt = last_dt + timedelta(days=float(med_interval))
            
            while next_dt < cutoff and (cutoff - next_dt).days <= med_interval * 1.8:
                next_dt += timedelta(days=float(med_interval))
                
            if cutoff <= next_dt <= cutoff_plus_90:
                next_dates[cat] = next_dt
                
        if not next_dates: preds[client_id] = 'none'
        else: preds[client_id] = min(next_dates.keys(), key=lambda k: next_dates[k])
            
    return preds

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'data')
    
    valid_clean_path = os.path.join(base_dir, 'clean_valid_transactions.csv')
    valid_preds = predict_recurring_merchant(valid_clean_path)
    
    valid_labels_path = os.path.join(base_dir, 'valid_labels.csv')
    valid_labels_df = pd.read_csv(valid_labels_path)
    valid_labels_df['pred'] = valid_labels_df['client_id'].map(valid_preds).fillna('none')
    
    classes = ['cloud', 'gym', 'insurance', 'mobile', 'music', 'software', 'streaming', 'none']
    f1_scores = []
    
    print(f'{"Class":12s} | {"Precision":10s} | {"Recall":10s} | {"F1-Score":10s}')
    print('-' * 52)
    for c in classes:
        tp = ((valid_labels_df['target_next_recurring_merchant'] == c) & (valid_labels_df['pred'] == c)).sum()
        fp = ((valid_labels_df['target_next_recurring_merchant'] != c) & (valid_labels_df['pred'] == c)).sum()
        fn = ((valid_labels_df['target_next_recurring_merchant'] == c) & (valid_labels_df['pred'] != c)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        print(f'{c:12s} | {prec:10.4f} | {rec:10.4f} | {f1:10.4f}')
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    print('-' * 52)
    print(f'FINAL MACRO-F1: {macro_f1:.4f}')
    
    print("\\n--- PHASE 3: Generation of Submission for Milestone 1 ---")
    test_clean_path = os.path.join(base_dir, 'clean_test_transactions.csv')
    test_preds = predict_recurring_merchant(test_clean_path)
    
    sample_sub_path = os.path.join(base_dir, 'sample_submission.csv')
    sub_df = pd.read_csv(sample_sub_path)
    
    sub_df['predicted_next_recurring_merchant'] = sub_df['client_id'].map(test_preds).fillna('none')
    
    output_sub_dir = os.path.join(script_dir, '..', 'submissions')
    os.makedirs(output_sub_dir, exist_ok=True)
    output_sub_path = os.path.join(output_sub_dir, 'submission_milestone1.csv')
    sub_df.to_csv(output_sub_path, index=False)
    
    print(f"Submission file saved in: {output_sub_path}")

if __name__ == '__main__':
    main()
