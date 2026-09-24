import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

def clean_text(text):
    """
    Pulizia del testo specifica per Embeddings.
    Rimuove date, numeri grezzi, ID transazione e caratteri speciali
    per lasciare solo la pura 'semantica' della spesa.
    """
    text = str(text).lower()
    # Rimuove date (es. 12/05/2026 o 12-05)
    text = re.sub(r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?', ' ', text)
    # Rimuove codici alfanumerici che contengono numeri (es. ID9938, tx123) o numeri isolati
    text = re.sub(r'\b\w*\d+\w*\b', ' ', text)
    # Rimuove caratteri speciali (tiene solo lettere e spazi)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "unknown"

def classify_transaction(mcc, desc, amt):
    desc_lower = desc.lower()
    noise = [
        'salary', 'atm withdrawal', 'fresh foods', 'pharmacy', 
        'hotel booking', 'electronics shop', 'coffee shop', 
        'neighborhood market', 'grocery store', 'online marketplace', 
        'ride share', 'casual dining', 'p2p send', 'p2p receive', 'service fee'
    ]
    # Invece di ritornare None (che ci farebbe scartare la riga), ritorniamo 'none' per tenerla
    if any(k in desc_lower for k in noise): return 'none'

    if mcc == '7997' or any(k in desc_lower for k in ['gym', 'fitness', 'fit club']): return 'gym'
    if mcc == '6300' or any(k in desc_lower for k in ['insurance', 'safe cover', 'policy', 'cover plan']): return 'insurance'
    if mcc == '4814' or any(k in desc_lower for k in ['phone', 'contract', 'telecom', 'carrier']): return 'mobile'
    
    if mcc == '5734':
        if any(k in desc_lower for k in ['cloud', 'storage', 'backup']): return 'cloud'
        return 'software'
    
    if mcc == '5732':
        if any(k in desc_lower for k in ['cloud', 'storage', 'backup', 'service plan']): return 'cloud'
    
    if mcc == '5812':
        if any(k in desc_lower for k in ['audio', 'member pass']): return 'music'
        if any(k in desc_lower for k in ['video', 'media stream', 'streaming', 'stream']): return 'streaming'
            
    if 'digital plus' in desc_lower or 'premium plan' in desc_lower:
        if mcc == '4814': return 'mobile'
        if mcc == '5734': return 'software'
        if mcc == '5812': return 'music' if amt < 16 else 'streaming'
                
    if 'monthly plan' in desc_lower:
        if mcc == '4814': return 'mobile'
        if mcc == '6300': return 'insurance'
        if mcc == '7997': return 'gym'
        if mcc == '5734': return 'software'
        if mcc == '5732': return 'cloud'
        if mcc == '5812': return 'streaming'
        
    return 'none'

def process_file(input_path, output_path):
    print(f"Inizio elaborazione per AI Embedding: {input_path}")
    records = []
    with open(input_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            
            # --- MODIFICA CHIAVE ---
            # NON scartiamo più NESSUNA transazione! (Né gli 'in', né i bancomat).
            # Tutto passa all'Intelligenza Artificiale, che imparerà le correlazioni temporali.
            # -----------------------
            
            cat = classify_transaction(d['mcc'], d['description'], d['amount'])
            
            records.append({
                'client_id': d['client_id'],
                'timestamp': d['timestamp'],
                'amount': d['amount'],
                'mcc': d['mcc'],
                'description': d['description'],
                'clean_description': clean_text(d['description']),  # <--- TESTO PULITO PER L'EMBEDDING
                'candidate_family': cat,
                'is_recurring_candidate': 1 if cat != 'none' else 0,
                'type': d['type'],
                'direction': d['direction'],
                'currency': d['currency']
            })

    if not records:
        print(f"Nessuna transazione utile in {input_path}")
        return

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    if df['timestamp'].dt.tz is not None:
        cutoff = datetime(2026, 1, 1, tzinfo=df['timestamp'].dt.tz)
    else:
        cutoff = datetime(2026, 1, 1)
        
    df['days_to_cutoff'] = (cutoff - df['timestamp']).dt.days

    # Ordinamento cronologico fondamentale per le serie temporali dell'AI
    df = df.sort_values(['client_id', 'timestamp']).reset_index(drop=True)

    df['days_since_prev_transaction'] = df.groupby('client_id')['timestamp'].diff().dt.days.fillna(0)

    mask = df['candidate_family'] != 'none'
    df_cand = df[mask].copy()
    
    if not df_cand.empty:
        group = df_cand.groupby(['client_id', 'candidate_family'])
        
        df_cand['days_since_prev_same_family'] = group['timestamp'].diff().dt.days.fillna(0)
        df_cand['count_same_family_before'] = group.cumcount()
        
        df_cand['gap'] = group['timestamp'].diff().dt.days
        df_cand['median_interval_same_family'] = group['gap'].expanding().median().reset_index(level=[0,1], drop=True).fillna(0)
        df_cand['std_interval_same_family'] = group['gap'].expanding().std().reset_index(level=[0,1], drop=True).fillna(0)
        
        df_cand['median_amount_same_family'] = group['amount'].expanding().median().reset_index(level=[0,1], drop=True)
        df_cand['amount_vs_family_median'] = df_cand['amount'] - df_cand['median_amount_same_family']
        
        cols_to_map = [
            'days_since_prev_same_family', 'count_same_family_before', 
            'median_interval_same_family', 'std_interval_same_family',
            'median_amount_same_family', 'amount_vs_family_median'
        ]
        for c in cols_to_map:
            df[c] = df.index.map(df_cand[c]).fillna(0)
    else:
        cols_to_map = [
            'days_since_prev_same_family', 'count_same_family_before', 
            'median_interval_same_family', 'std_interval_same_family',
            'median_amount_same_family', 'amount_vs_family_median'
        ]
        for c in cols_to_map:
            df[c] = 0

    order = [
        'client_id', 'timestamp', 'amount', 'mcc', 'description', 'clean_description',
        'candidate_family', 'is_recurring_candidate', 'type', 'direction', 'currency',
        'day_of_week', 'day_of_month', 'month', 'days_to_cutoff',
        'days_since_prev_transaction', 'days_since_prev_same_family', 'count_same_family_before',
        'median_interval_same_family', 'std_interval_same_family', 'median_amount_same_family', 'amount_vs_family_median'
    ]
    df = df[order]

    df.to_csv(output_path, index=False)
    print(f"-> Feature Engineering completato: {output_path} ({len(df)} righe, {len(df.columns)} colonne)")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', 'data')
    
    files = [
        ('train_transactions.jsonl', 'clean_train_transactions.csv'),
        ('valid_transactions.jsonl', 'clean_valid_transactions.csv'),
        ('test_transactions.jsonl', 'clean_test_transactions.csv')
    ]
    for in_f, out_f in files:
        ip = os.path.join(base_dir, in_f)
        op = os.path.join(base_dir, out_f)
        if os.path.exists(ip): process_file(ip, op)
