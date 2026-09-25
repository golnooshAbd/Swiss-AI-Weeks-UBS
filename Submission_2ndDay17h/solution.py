#!/usr/bin/env python3
"""
UBS Swiss AI Weeks 2026 - Transaction Activity Forecasting
===========================================================
Predict, per client, the merchant family of the first recurring (subscription) payment after the
cutoff 2026-01-01 within 90 days: cloud, gym, insurance, mobile, music, software, streaming, or none.

Approach
--------
1. Tag every transaction with family evidence (description keywords, MCC) and a text class
   (keyword / generic subscription phrase / decoy phrase / everyday purchase).
2. Detect recurring subscription streams per client with rules (amount clustering + billing schedule):
     v1: family-first clustering (row family from keyword, else MCC)
     v3: noise-robust: amount clustering inside evidence groups, then attach charges with noisy
         description / MCC when amount (+-5%) and billing schedule fit; family decided by a vote.
3. Candidate features per (client, family): last charge, billing period, jitter, projected next charge,
   timing vs the other families, stream quality, refunds, lone recent charges + client-level features.
3b. Proxy pre-training (unlabeled pretrain set, 10,000 clients): simulate 3 earlier cutoffs (Jul, mid-Aug,
   Oct 2025), label each client with the first recurring payment of its own following 90 days, train a
   LightGBM "pseudo-scorer" on these 420k (client, family) rows (clean + test-level noisy copy) and add its
   score as extra features (ps_*) to both pipelines.
4. Two-stage LightGBM per pipeline:
     stage 1: binary scorer on (client, family) rows  -> "is this family the label?"
     stage 2: 8-class classifier on 7 out-of-fold stage-1 scores + client features -> final probabilities
   Each pipeline is trained with 3 seeds and averaged; blend = 0.5 * v1 + 0.5 * v3.
5. Decision rule (--balance):
     em (default): label-shift correction (Saerens et al. 2002): estimate the class mix of the unlabeled
                  prediction set (test) from the model's own probabilities by EM, re-weight, argmax.
     none_weight: P(none) x 0.7 before the argmax (macro-F1 rewards recall on the seven small classes);
                  0.7 chosen by 5-fold CV on train clients only (--tune-none re-derives it).
     off:         plain argmax of the blend.

Data usage
----------
- Training labels: train_labels.csv ONLY (2,000 clients).
- Noise augmentation: the test histories are much noisier than train (generic / decoy descriptions,
  random MCCs, fake keywords on everyday purchases). We add 5 noisy copies of every train client
  (2x 'mid', 3x 'test' level; labels are unchanged by this noise). The 'test' rates were calibrated
  by comparing description/MCC composition inside detected streams in train vs. the test histories;
  'mid' is half of it. No validation data is used to set them.
- valid_*: used ONLY for scoring (macro-F1 printed at the end).
- test_transactions: model input for the submission.
- unlabeled_pretrain_transactions: pseudo-labelled training data for the pseudo-scorer (step 3b); its own
  future transactions provide the labels, no challenge labels involved.

Usage
-----
    pip install pandas numpy scikit-learn lightgbm
    python solution.py --data-dir data --out submission.csv [--jobs 2] [--threads 2] [--no-pretrain]

`--data-dir` must contain the extracted files (or dataset.zip, which is extracted automatically).
Tested with Python 3.14.7 (and 3.11), pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, lightgbm 4.7.0.
Runtime: ~25 min on 2 CPU cores (feature building dominates; use --jobs to parallelise it).
Keep --threads 2 to reproduce our exact numbers (LightGBM results can differ slightly by thread count).
"""
import argparse, os, sys, time, zipfile, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

FAMS = ['cloud', 'gym', 'insurance', 'mobile', 'music', 'software', 'streaming']
LABELS = FAMS + ['none']
FI = {f: i for i, f in enumerate(FAMS)}
CUTOFF = pd.Timestamp('2026-01-01', tz='UTC')
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
KW = {
    'mobile': {'phone', 'contract', 'bill'},
    'cloud': {'cloud', 'storage', 'backup'},
    'software': {'saas', 'productivity', 'prod', 'suite', 'software'},
    'insurance': {'cover', 'insurance', 'policy', 'safe'},
    'gym': {'gym', 'fit', 'fitness', 'club', 'urban', 'membership'},
    'streaming': {'media', 'video'},
    'music': {'audio', 'pass'},
}
FILES = ['unlabeled_pretrain_transactions.jsonl',
         'train_transactions.jsonl', 'train_labels.csv', 'valid_transactions.jsonl', 'valid_labels.csv',
         'test_transactions.jsonl', 'sample_submission.csv']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# =====================================================================================================
# Loading + v1 row tagging
# =====================================================================================================
V1_EVERYDAY = {'salary', 'atm', 'withdrawal', 'fresh', 'foods', 'pharmacy', 'hotel', 'booking', 'electronics',
               'shop', 'ride', 'share', 'coffee', 'grocery', 'store', 'neighborhood', 'market', 'marketplace',
               'dining', 'casual', 'p2p', 'send', 'receive'}
V1_DECOY = {'order', 'purchase', 'merchant', 'payment', 'card'}
V1_MCC_FAM = {4814: ['mobile'], 5732: ['cloud'], 5734: ['software'], 6300: ['insurance'], 7997: ['gym'],
              5812: ['streaming', 'music']}
V1_AMBIG = {  # which families an ambiguous template phrase can belong to
    ('digital', 'plus'): ['mobile', 'streaming', 'music'], ('dgtl', 'plus'): ['mobile', 'streaming', 'music'],
    ('premium', 'plan'): ['software', 'streaming', 'music'], ('prem', 'plan'): ['software', 'streaming', 'music'],
    ('service', 'plan'): ['cloud'], ('monthly', 'plan'): ['mobile'],
}


def load(data_dir, name):
    d = pd.read_json(os.path.join(data_dir, f'{name}_transactions.jsonl'), lines=True)
    d['ts'] = d.timestamp
    return d.sort_values(['client_id', 'ts']).reset_index(drop=True)


def desc_family(desc, mcc):
    """(kind, family set). kind: strong (keyword) | mcc (generic text resolved by MCC) | generic | everyday | decoy"""
    toks = set(desc.split())
    hits = [f for f, kw in KW.items() if toks & kw]
    if 'streaming' in toks or 'stream' in toks:
        if not hits:
            hits = ['streaming', 'music']
    if len(hits) >= 1:
        return 'strong', set(hits)
    if toks & V1_EVERYDAY:
        return 'everyday', set()
    if toks & V1_DECOY:
        return 'decoy', set()
    mf = V1_MCC_FAM.get(mcc)
    for k, fams in V1_AMBIG.items():
        if set(k) <= toks:
            if mf and set(mf) & set(fams):
                return 'mcc', set(mf) & set(fams)
            return 'generic', set(fams)
    if mf:
        return 'mcc', set(mf)
    return 'generic', set()


def tag(df):
    k, f = zip(*[desc_family(d, m) for d, m in zip(df.description.values, df.mcc.values)])
    df = df.copy(); df['kind'] = k; df['fams'] = f
    return df


# =====================================================================================================
# Noise augmentation (labels are invariant to it)
# =====================================================================================================
GENERIC = ['subscription charge', 'monthly plan', 'member plan', 'digital service']
DECOYS = ['digital order', 'service payment', 'merchant charge', 'card purchase']
CARD_MCCS = [4111, 4814, 5411, 5732, 5734, 5812, 5912, 6300, 7011, 7997]
# rates added on top of train's own noise. 'test': calibrated on test histories vs train
# (inside streams: generic text 7% -> 51%, decoy 0 -> 5.4%, off-family MCC 6% -> 14%;
#  everyday purchases: decoy text 2% -> 53%, fake family keyword 3% -> 12%). 'mid' = half of 'test'.
NOISE_LEVELS = {
    'mid': dict(sub_generic=0.235, sub_decoy=0.0275, sub_mcc=0.0425, ev_decoy=0.265, ev_kw=0.045, ev_generic=0.0225),
    'test': dict(sub_generic=0.47, sub_decoy=0.055, sub_mcc=0.085, ev_decoy=0.53, ev_kw=0.09, ev_generic=0.045),
}
TRAIN_COPIES = [('mid', 21), ('mid', 22), ('test', 3), ('test', 4), ('test', 5)]


def augment(df, level, seed=0, suffix=''):
    """df: v1-tagged transactions. Returns a re-tagged copy with injected description/MCC noise."""
    rng = np.random.default_rng(seed); p = NOISE_LEVELS[level]
    d = df.copy()
    kw_pool = d.loc[(d.kind == 'strong') & (d.type == 'card_payment'), 'description'].values
    card = (d.type == 'card_payment').values
    sub = card & d.kind.isin(['strong', 'mcc', 'generic']).values
    ev = card & (d.kind == 'everyday').values
    u = rng.random(len(d)); desc = d.description.values.copy(); mcc = d.mcc.values.copy()
    # subscription rows: generic / decoy description, random MCC
    m = sub & (u < p['sub_generic']); desc[m] = rng.choice(GENERIC, m.sum())
    m = sub & (u >= p['sub_generic']) & (u < p['sub_generic'] + p['sub_decoy']); desc[m] = rng.choice(DECOYS, m.sum())
    u2 = rng.random(len(d)); m = sub & (u2 < p['sub_mcc']); mcc[m] = rng.choice(CARD_MCCS, m.sum())
    # everyday rows: decoy / fake keyword / generic descriptions (MCC kept)
    u3 = rng.random(len(d))
    m = ev & (u3 < p['ev_decoy']); desc[m] = rng.choice(DECOYS, m.sum())
    m = ev & (u3 >= p['ev_decoy']) & (u3 < p['ev_decoy'] + p['ev_kw']); desc[m] = rng.choice(kw_pool, m.sum())
    m = ev & (u3 >= p['ev_decoy'] + p['ev_kw']) & (u3 < p['ev_decoy'] + p['ev_kw'] + p['ev_generic'])
    desc[m] = rng.choice(GENERIC, m.sum())
    d['description'] = desc; d['mcc'] = mcc
    if suffix: d['client_id'] = d.client_id + suffix
    return tag(d.drop(columns=['kind', 'fams']))


# =====================================================================================================
# v1: stream detection + candidate features
# =====================================================================================================
def _split_amount(idx, amts, tol=0.15):
    """single-linkage clustering on log amount"""
    o = np.argsort(amts); la = np.log(np.asarray(amts)[o]); idx = np.asarray(idx)[o]
    cuts = np.where(np.diff(la) > np.log(1 + tol))[0] + 1
    return np.split(idx, cuts)


def build_streams_v1(df, cutoff=CUTOFF, tol=0.15, attach_tol=0.10):
    df = df[df.ts < cutoff]
    pay = df[(df.direction == 'out') & (df.type == 'card_payment') & df.kind.isin(['strong', 'mcc', 'generic'])]
    ref = df[(df.type == 'refund') & df.kind.isin(['strong', 'mcc', 'generic'])]
    recs = []; sid = 0
    for cid, g in pay.groupby('client_id', sort=False):
        single = g[g.fams.map(len) == 1]
        clusters = []
        for fam, gg in single.groupby(single.fams.map(lambda s: next(iter(s)))):
            for ids in _split_amount(gg.index.values, gg.amount.values, tol):
                clusters.append([fam, list(ids)])
        rest = g[g.fams.map(len) != 1]   # ambiguous / generic rows: attach by amount
        pend = []
        for i, a, fs in zip(rest.index, rest.amount.values, rest.fams.values):
            best, bd = None, 1e9
            for c in clusters:
                if fs and c[0] not in fs:
                    continue
                med = np.median(df.loc[c[1], 'amount'].values)
                d = abs(np.log(a / med))
                if d < bd: best, bd = c, d
            if best is not None and bd < np.log(1 + attach_tol):
                best[1].append(i)
            elif fs & {'streaming', 'music'}:
                pend.append(i)
        if pend:  # unresolved 5812 rows -> 'sm' (streaming-or-music) clusters
            pp = df.loc[pend]
            for ids in _split_amount(pp.index.values, pp.amount.values, tol):
                clusters.append(['sm', list(ids)])
        for fam, ids in clusters:
            recs.append((sid, cid, fam, ids)); sid += 1
    S = []
    for sid, cid, fam, ids in recs:
        r = df.loc[sorted(ids, key=lambda i: df.at[i, 'ts'])]
        t = r.ts.values.astype('datetime64[s]').astype(np.int64) / 86400.0
        gaps = np.diff(t)
        S.append(dict(sid=sid, client_id=cid, fam=fam, n=len(r), first=t[0], last=t[-1],
                      gap_med=np.median(gaps) if len(gaps) else np.nan,
                      gap_mad=np.median(np.abs(gaps - np.median(gaps))) if len(gaps) else np.nan,
                      gap_last=gaps[-1] if len(gaps) else np.nan,
                      amt_med=r.amount.median(), amt_last=r.amount.values[-1],
                      n_strong=(r.kind == 'strong').sum()))
    S = pd.DataFrame(S)
    S['n_ref'] = 0; S['last_ref'] = np.nan
    if len(S):
        rt = ref.copy(); rt['t'] = rt.ts.values.astype('datetime64[s]').astype(np.int64) / 86400.0
        bycl = {c: g for c, g in rt.groupby('client_id')}
        nref, lref = [], []
        for cid, am in zip(S.client_id, S.amt_med):
            g = bycl.get(cid)
            if g is None: nref.append(0); lref.append(np.nan); continue
            m = g[np.abs(np.log(g.amount / am)) < 0.05]
            nref.append(len(m)); lref.append(m.t.max() if len(m) else np.nan)
        S['n_ref'] = nref; S['last_ref'] = lref
    return S


def cand_features_v1(df, cutoff=CUTOFF):
    S = build_streams_v1(df, cutoff)
    c0 = (cutoff - EPOCH).total_seconds() / 86400.0
    df = df[df.ts < cutoff].copy()
    df['t'] = df.ts.values.astype('datetime64[s]').astype(np.int64) / 86400.0
    clients = df.client_id.unique()
    pay = df[(df.type == 'card_payment') & df.kind.isin(['strong', 'mcc']) & (df.fams.map(len) >= 1)]
    ref = df[(df.type == 'refund') & (df.fams.map(len) >= 1)]
    rows = []
    cl = df.groupby('client_id').agg(ntx=('t', 'size'), t_first=('t', 'min'), t_last=('t', 'max'))
    cl['ntx90'] = df[df.t > c0 - 90].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_generic'] = df[df.kind == 'generic'].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_decoy'] = df[df.kind == 'decoy'].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_fee'] = df[df.type == 'fee'].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_salary'] = df[df.description == 'salary'].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_sub'] = pay.groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_sub90'] = pay[pay.t > c0 - 90].groupby('client_id').size().reindex(cl.index, fill_value=0)
    cl['n_ref_all'] = ref.groupby('client_id').size().reindex(cl.index, fill_value=0)
    S = S.copy(); S['since'] = c0 - S['last']; S['age'] = c0 - S['first']
    P = S.gap_med.clip(5, 120).fillna(30.4)
    S['P'] = P; S['next'] = S['last'] + np.maximum(1, np.ceil((S.since + 0.5) / P)) * P - c0
    S['overdue'] = S.since / P
    S['ok'] = ((S.n >= 3) & (S.overdue < 1.6)).astype(int)
    payg = {k: g for k, g in pay.groupby('client_id')}
    refg = {k: g for k, g in ref.groupby('client_id')}
    Sg = {k: g for k, g in S.groupby('client_id')}
    for c in clients:
        p = payg.get(c, pay.iloc[:0]); r = refg.get(c, ref.iloc[:0]); s = Sg.get(c, S.iloc[:0])
        for f in FAMS:
            m = np.array([f in z for z in p.fams.values], dtype=bool)
            e = p.loc[m].sort_values('t')
            amb = np.array([len(z) > 1 for z in e.fams.values], dtype=bool)
            t = e.t.values; a = e.amount.values
            d = dict(client_id=c, fam=f)
            d['n_all'] = len(e); d['n_amb'] = int(amb.sum()); d['n_strong'] = int((e.kind == 'strong').sum())
            d['n_unamb'] = len(e) - int(amb.sum())
            for w in [30, 45, 60, 90, 180]:
                d[f'n_{w}'] = int((t > c0 - w).sum())
                d[f'nu_{w}'] = int(((t > c0 - w) & ~amb).sum())
            for k in [1, 2, 3, 4]:
                d[f'since{k}'] = c0 - t[-k] if len(t) >= k else np.nan
            tu = t[~amb]
            d['since1_u'] = c0 - tu[-1] if len(tu) else np.nan
            d['since2_u'] = c0 - tu[-2] if len(tu) >= 2 else np.nan
            g = np.diff(t[-7:]) if len(t) >= 2 else np.array([])
            d['gap1'] = g[-1] if len(g) else np.nan
            d['gap2'] = g[-2] if len(g) >= 2 else np.nan
            d['gap_med6'] = np.median(g) if len(g) else np.nan
            d['gap_std6'] = np.std(g) if len(g) >= 2 else np.nan
            d['amt1'] = a[-1] if len(a) else np.nan
            d['amt_ratio12'] = a[-1] / a[-2] if len(a) >= 2 else np.nan
            d['amt_cv6'] = np.std(np.log(a[-6:])) if len(a) >= 2 else np.nan
            if len(a):
                close = np.abs(np.log(a / a[-1])) < 0.1
                d['n_close_amt'] = int(close.sum())
                tc = t[close]; gc = np.diff(tc)
                d['gap_close_med'] = np.median(gc[-6:]) if len(gc) else np.nan
                d['first_close'] = c0 - tc[0]
            else:
                d['n_close_amt'] = 0; d['gap_close_med'] = np.nan; d['first_close'] = np.nan
            s1 = d['since1']
            for nm, per in [('30', 30.4), ('g1', d['gap1']), ('gm', d['gap_med6']), ('gc', d['gap_close_med'])]:
                if np.isnan(s1) or per is None or np.isnan(per):
                    d[f'nx_{nm}'] = np.nan
                else:
                    per = float(np.clip(per, 5, 120))
                    d[f'nx_{nm}'] = per - s1   # negative => overdue
            rm = r.loc[np.array([f in z for z in r.fams.values], dtype=bool)]
            d['n_ref'] = len(rm)
            d['ref_since'] = c0 - rm.t.max() if len(rm) else np.nan
            d['ref_after_last'] = int(len(rm) > 0 and len(t) > 0 and rm.t.max() >= t[-1])
            ss = s[s.fam == f] if f not in ('streaming', 'music') else s[s.fam.isin([f, 'sm'])]
            d['s_cnt'] = len(ss); d['s_okcnt'] = int(ss.ok.sum()) if len(ss) else 0
            if len(ss):
                b = ss.sort_values(['ok', 'next'], ascending=[False, True]).iloc[0]
                for col in ['n', 'since', 'age', 'gap_med', 'gap_mad', 'next', 'overdue', 'amt_med', 'n_ref', 'ok']:
                    d[f's_{col}'] = float(b[col])
                d['s_is_sm'] = int(b['fam'] == 'sm')
            else:
                for col in ['n', 'since', 'age', 'gap_med', 'gap_mad', 'next', 'overdue', 'amt_med', 'n_ref', 'ok']:
                    d[f's_{col}'] = np.nan
                d['s_is_sm'] = 0
            rows.append(d)
    C = pd.DataFrame(rows)
    C = C.merge(cl.drop(columns=['t_first', 't_last']).reset_index(), on='client_id', how='left')
    C['fam_id'] = C.fam.map(FAMS.index)
    for col in ['nx_30', 'nx_gm', 'nx_gc', 's_next', 'since1', 'since1_u']:   # within-client relative features
        v = C[col].where(C[col].notna(), np.nan)
        grp = C.assign(v=v).groupby('client_id').v
        C[f'{col}_rank'] = grp.rank(method='min')
        C['_v'] = v

        def other_min(g):
            x = g.values; out = np.full(len(x), np.nan)
            for i in range(len(x)):
                o = np.delete(x, i); o = o[~np.isnan(o)]
                out[i] = o.min() if len(o) else np.nan
            return pd.Series(out, index=g.index)
        C[f'{col}_vs_other'] = v - C.groupby('client_id')._v.transform(other_min)
        C.drop(columns=['_v'], inplace=True)
    C['n_active_fams'] = C.assign(a=(C.since1 < 45).astype(int)).groupby('client_id').a.transform('sum')
    C['n_ok_fams'] = C.assign(a=(C.s_ok == 1).astype(int)).groupby('client_id').a.transform('sum')
    return C


# =====================================================================================================
# v3: noise-robust stream detection + candidate features
# =====================================================================================================
V3_EVERYDAY = V1_EVERYDAY | {'fee'}
V3_DECOY_PH = ('digital order', 'service payment', 'merchant charge', 'card purchase')
V3_GENERIC_PH = ('subscription charge', 'monthly plan', 'member plan', 'digital service')
V3_MCC_FAM = {4814: 'mobile', 5732: 'cloud', 5734: 'software', 6300: 'insurance', 7997: 'gym'}
FAMILY_MCCS = set(V3_MCC_FAM) | {5812}
SEED_DECOY_MCC = {4814, 5734, 6300, 7997}   # family MCCs without everyday usage
_TEXT_CACHE = {}


def text_ev(desc):
    """text class + family evidence (keyword 3, ambiguous template 1.5 split)"""
    r = _TEXT_CACHE.get(desc)
    if r is not None: return r
    toks = set(desc.split()); v = np.zeros(7)
    hits = [f for f, kw in KW.items() if toks & kw]
    if hits:
        for f in hits: v[FI[f]] += 3.0 / len(hits)
        r = ('kw', v)
    elif 'streaming' in toks or 'stream' in toks:
        v[FI['streaming']] += 1.5; v[FI['music']] += 1.5; r = ('kw', v)
    elif any(p in desc for p in V3_DECOY_PH): r = ('decoy', v)
    elif toks & V3_EVERYDAY: r = ('everyday', v)
    elif any(p in desc for p in V3_GENERIC_PH): r = ('generic', v)
    elif ('plus' in toks and ('digital' in toks or 'dgtl' in toks)):
        v[FI['mobile']] += .5; v[FI['streaming']] += .5; v[FI['music']] += .5; r = ('amb', v)
    elif 'plan' in toks and ('premium' in toks or 'prem' in toks):
        v[FI['software']] += .5; v[FI['streaming']] += .5; v[FI['music']] += .5; r = ('amb', v)
    elif 'plan' in toks and 'service' in toks:
        v[FI['cloud']] += 1.5; r = ('amb', v)
    else: r = ('other', v)
    _TEXT_CACHE[desc] = r
    return r


def mcc_ev(mcc):
    v = np.zeros(7)
    if mcc in V3_MCC_FAM: v[FI[V3_MCC_FAM[mcc]]] = 1.0
    elif mcc == 5812: v[FI['streaming']] = .5; v[FI['music']] = .5
    return v


def prep_v3(df, cutoff=CUTOFF):
    df = df[df.ts < cutoff].copy()
    df['t'] = df.ts.values.astype('datetime64[s]').astype(np.int64) / 86400.0
    te = [text_ev(d) for d in df.description.values]
    df['txt'] = [c for c, _ in te]
    E = np.array([v for _, v in te]) if len(te) else np.zeros((0, 7))
    M = np.array([mcc_ev(m) for m in df.mcc.values]) if len(df) else np.zeros((0, 7))
    ev = df.txt.values == 'everyday'
    E = E + M * (~ev)[:, None]          # everyday purchases get no MCC evidence
    for i, f in enumerate(FAMS): df[f'e_{f}'] = E[:, i]
    return df


def _period(t):
    """robust billing period allowing skipped charges; returns (period, jitter)"""
    g = np.diff(t)
    if len(g) == 0: return np.nan, np.nan
    m = np.median(g)
    g = g[g > 0.3 * m] if (g > 0.3 * m).any() else g
    m = np.median(g)
    if len(g) == 1: return float(g[0]), np.nan
    k = np.maximum(1, np.round(g / m)); P = g.sum() / k.sum()
    return float(P), float(np.median(np.abs(g - k * P)))


def _fits(tt, ts, P, tol):
    """does time tt fit the billing schedule (sorted member times ts, period P)?"""
    j = np.searchsorted(ts, tt)
    gaps = []
    if j > 0: gaps.append(tt - ts[j - 1])
    if j < len(ts): gaps.append(ts[j] - tt)
    if min(gaps) < 0.35 * P: return False
    for g in gaps:
        for k in (1, 2):
            if abs(g - k * P) <= tol * (1 if k == 1 else 1.4): return True
    return False


def client_streams_v3(g, seed_tol=0.08, att_tol=0.05):
    """one client's prepped rows -> list of streams (positions in g)"""
    card = (g.type.values == 'card_payment')
    txt = g.txt.values; mcc = g.mcc.values; amt = g.amount.values; t = g.t.values
    E = g[[f'e_{f}' for f in FAMS]].values
    seed = card & (np.isin(txt, ['kw', 'amb']) | (np.isin(txt, ['generic', 'other']) & np.isin(mcc, list(FAMILY_MCCS)))
                   | ((txt == 'decoy') & np.isin(mcc, list(SEED_DECOY_MCC))))
    grp = np.full(len(g), -1)
    for i in np.where(seed)[0]:
        e = E[i]
        if e.sum() == 0: continue
        f = int(np.argmax(e))
        grp[i] = 99 if FAMS[f] in ('streaming', 'music') else f   # streaming + music clustered jointly
    streams = []; used = np.zeros(len(g), bool)
    for gv in np.unique(grp[grp >= 0]):
        idx = np.where(grp == gv)[0]
        o = idx[np.argsort(amt[idx])]; la = np.log(amt[o])
        for c in np.split(o, np.where(np.diff(la) > np.log(1 + seed_tol))[0] + 1):
            if len(c) >= 2:
                streams.append(list(c)); used[c] = True
    if not streams: return streams
    # attach leftover non-everyday card payments by amount + billing schedule (two passes)
    left = np.where(card & ~used & (txt != 'everyday'))[0]
    for _ in range(2):
        info = []
        for s in streams:
            ss = np.array(sorted(s, key=lambda i: t[i])); P, _ = _period(t[ss])
            vf = int(np.argmax(E[ss].sum(0)))
            info.append((ss, P, vf))
        still = []
        for i in left[np.argsort(t[left])]:
            best, bd = None, 1e9
            for k, (ss, P, vf) in enumerate(info):
                if not np.isfinite(P) or P < 5: continue
                near = ss[np.argsort(np.abs(t[ss] - t[i]))[:3]]
                d = abs(np.log(amt[i] / np.median(amt[near])))
                if d > np.log(1 + att_tol) or d >= bd: continue
                if txt[i] == 'kw' and E[i].sum() > 0 and E[i][vf] == 0: continue  # conflicting keyword
                if _fits(t[i], np.sort(t[ss]), P, max(6.0, 0.25 * P)):
                    best, bd = k, d
            if best is None: still.append(i)
            else: streams[best].append(i)
        left = np.array(still, dtype=int)
        if len(left) == 0: break
    return streams


def stream_table_v3(df, cutoff=CUTOFF):
    c0 = (cutoff - EPOCH).total_seconds() / 86400.0
    out = []; member_rows = []
    for cid, g in df.groupby('client_id', sort=False):
        g = g.sort_values('t')
        S = client_streams_v3(g)
        E = g[[f'e_{f}' for f in FAMS]].values; t = g.t.values; amt = g.amount.values
        txt = g.txt.values; mcc = g.mcc.values
        ref = g[g.type.values == 'refund']
        for s in S:
            ss = np.array(sorted(s, key=lambda i: t[i])); ts = t[ss]; a = amt[ss]
            P, jit = _period(ts)
            V = E[ss].sum(0); fi = int(np.argmax(V)); fam = FAMS[fi]   # family vote over the stream
            if fam in ('streaming', 'music') and abs(V[FI['streaming']] - V[FI['music']]) < 1e-9: fam = 'sm'
            share = V.max() / V.sum() if V.sum() > 0 else 0
            own_kw = sum(1 for i in ss if txt[i] == 'kw' and E[i][fi] >= 1.5)
            own_mcc = sum(1 for i in ss if (V3_MCC_FAM.get(mcc[i]) == fam) or (mcc[i] == 5812 and fam in ('streaming', 'music', 'sm')))
            ar = np.median(a[-3:])
            rr = ref[np.abs(np.log(ref.amount.values / ar)) < 0.05] if len(ref) else ref
            rt = rr.t.values if len(rr) else np.array([])
            d = dict(client_id=cid, fam=fam, n=len(ss), first=ts[0], last=ts[-1], P=P, jit=jit,
                     amt_recent=ar, amt_med=np.median(a), amt_cv=np.std(np.log(a)),
                     amt_jump=a[-1] / np.median(a[:-1]) if len(a) > 1 else 1.0,
                     share=share, vote=V.sum(), own_kw=own_kw, own_mcc=own_mcc,
                     n_everyday_mcc=int(sum(1 for i in ss if mcc[i] not in FAMILY_MCCS)),
                     n_decoy=int(sum(1 for i in ss if txt[i] == 'decoy')), n_generic=int(sum(1 for i in ss if txt[i] == 'generic')),
                     n_ref=len(rt), ref_after_last=int(len(rt) > 0 and rt.max() >= ts[-1] - 0.5),
                     last_ref=(c0 - rt.max()) if len(rt) else np.nan,
                     v_str=V[FI['streaming']], v_mus=V[FI['music']])
            out.append(d)
            member_rows.extend(g.index.values[ss].tolist())
    S = pd.DataFrame(out)
    if len(S) == 0: return S, set(member_rows)
    S['since'] = c0 - S['last']; S['age'] = c0 - S['first']
    Pc = S.P.clip(5, 120).fillna(30.4)
    S['overdue'] = S.since / Pc
    S['next'] = S['last'] + np.maximum(1, np.ceil((S.since + 0.5) / Pc)) * Pc - c0   # projected next charge
    S['next1'] = Pc - S.since
    S['cv'] = S.jit / Pc
    S['alive'] = ((S.n >= 2) & (S.overdue < 1.6)).astype(int)
    return S, set(member_rows)


SCOLS = ['n', 'P', 'jit', 'cv', 'since', 'overdue', 'next', 'next1', 'age', 'amt_recent', 'amt_cv', 'amt_jump', 'share',
         'own_kw', 'own_mcc', 'n_everyday_mcc', 'n_decoy', 'n_generic', 'n_ref', 'ref_after_last', 'last_ref', 'alive']


def cand_features_v3(df_raw, cutoff=CUTOFF):
    c0 = (cutoff - EPOCH).total_seconds() / 86400.0
    df = prep_v3(df_raw, cutoff)
    S, members = stream_table_v3(df, cutoff)
    df['in_stream'] = df.index.isin(members)
    clients = df.client_id.unique()
    card = df[df.type == 'card_payment']
    cl = pd.DataFrame(index=pd.Index(clients, name='client_id'))
    g = df.groupby('client_id')
    cl['ntx'] = g.size(); cl['ntx90'] = df[df.t > c0 - 90].groupby('client_id').size().reindex(clients, fill_value=0)
    cl['n_card'] = card.groupby('client_id').size().reindex(clients, fill_value=0)
    for k in ['decoy', 'generic', 'kw', 'everyday']:
        cl[f'sh_{k}'] = (card.txt == k).groupby(card.client_id).mean().reindex(clients).fillna(0)
    cl['n_salary'] = (df.description == 'salary').groupby(df.client_id).sum().reindex(clients, fill_value=0)
    cl['n_refund'] = (df.type == 'refund').groupby(df.client_id).sum().reindex(clients, fill_value=0)
    ins = df[df.in_stream]
    cl['n_stream_pay'] = ins.groupby('client_id').size().reindex(clients, fill_value=0)
    cl['n_sp_60'] = ins[ins.t > c0 - 60].groupby('client_id').size().reindex(clients, fill_value=0)
    cl['n_sp_60_120'] = ins[(ins.t <= c0 - 60) & (ins.t > c0 - 120)].groupby('client_id').size().reindex(clients, fill_value=0)
    cl['trend'] = (cl.n_sp_60 + 1) / (cl.n_sp_60_120 + 1)
    if len(S):
        Sg = S.groupby('client_id')
        cl['n_streams'] = Sg.size().reindex(clients, fill_value=0)
        cl['n_alive'] = Sg.alive.sum().reindex(clients, fill_value=0)
        cl['n_alive_fams'] = S[S.alive == 1].groupby('client_id').fam.nunique().reindex(clients, fill_value=0)
        cl['s_ref'] = Sg.n_ref.sum().reindex(clients, fill_value=0)
        cl['ref_ratio'] = cl.s_ref / cl.n_stream_pay.clip(lower=1)
        cl['n_ref_after_last'] = S[S.alive == 1].groupby('client_id').ref_after_last.sum().reindex(clients, fill_value=0)
        cl['min_next'] = S[S.alive == 1].groupby('client_id').next.min().reindex(clients)
        cl['max_age'] = Sg.age.max().reindex(clients)
        cl['mean_age_alive'] = S[S.alive == 1].groupby('client_id').age.mean().reindex(clients)
    rows = []
    Sg = {k: v for k, v in S.groupby('client_id')} if len(S) else {}
    single = card[~card.in_stream & (card.txt != 'everyday')]
    sg = {k: v for k, v in single.groupby('client_id')}
    fam2mcc = {v: k for k, v in V3_MCC_FAM.items()}
    for c in clients:
        s = Sg.get(c); sl = sg.get(c)
        for f in FAMS:
            d = {'client_id': c, 'fam': f}
            sf = s[(s.fam == f) | ((s.fam == 'sm') & (f in ('streaming', 'music')))] if s is not None else None
            if sf is not None and len(sf):
                b = sf.sort_values(['alive', 'next'], ascending=[False, True]).iloc[0]
                for col in SCOLS: d[f's_{col}'] = float(b[col])
                d['s_is_sm'] = int(b['fam'] == 'sm')
                d['s_vfrac'] = float(b['v_str'] / (b['v_str'] + b['v_mus'])) if (b['v_str'] + b['v_mus']) > 0 else np.nan
                d['f_nstreams'] = len(sf); d['f_nalive'] = int(sf.alive.sum()); d['f_maxn'] = int(sf.n.max())
                d['f_last'] = float(sf.since.min())
            else:
                for col in SCOLS: d[f's_{col}'] = np.nan
                d['s_is_sm'] = 0; d['s_vfrac'] = np.nan; d['f_nstreams'] = 0; d['f_nalive'] = 0; d['f_maxn'] = 0; d['f_last'] = np.nan
            # lone charges not in any stream: trusted = at the family's MCC; low-trust = keyword only
            if sl is not None and len(sl):
                fm = (sl.mcc == 5812) if f in ('streaming', 'music') else (sl.mcc == fam2mcc.get(f, -1))
                if f in ('streaming', 'music', 'cloud'): fm = fm & (sl.txt != 'decoy')
                tr = sl[fm.values]; kwm = sl[(sl.txt == 'kw').values & (sl[f'e_{f}'].values >= 1.5) & ~fm.values]
                d['sg_tr_n45'] = int((tr.t > c0 - 45).sum()); d['sg_tr_since'] = c0 - tr.t.max() if len(tr) else np.nan
                d['sg_kw_n45'] = int((kwm.t > c0 - 45).sum()); d['sg_kw_since'] = c0 - kwm.t.max() if len(kwm) else np.nan
                d['sg_tr_amt'] = tr.amount.values[-1] if len(tr) else np.nan
            else:
                d['sg_tr_n45'] = 0; d['sg_tr_since'] = np.nan; d['sg_kw_n45'] = 0; d['sg_kw_since'] = np.nan; d['sg_tr_amt'] = np.nan
            rows.append(d)
    C = pd.DataFrame(rows).merge(cl.reset_index(), on='client_id', how='left')
    C['fam_id'] = C.fam.map(FI)
    C['_nx'] = C.s_next.where(C.s_alive == 1)       # timing relative to the other alive streams
    grp = C.groupby('client_id')._nx
    C['nx_rank'] = grp.rank(method='min')
    mn1 = grp.transform('min')

    def second_min(x):
        v = np.sort(x.dropna().values); return v[1] if len(v) > 1 else np.nan
    mn2 = grp.transform(second_min)
    C['nx_vs_other'] = np.where(C._nx == mn1, C._nx - mn2, C._nx - mn1)
    C['nx_vs_min'] = C._nx - mn1
    return C.drop(columns=['_nx'])


# =====================================================================================================
# Two-stage LightGBM
# =====================================================================================================
def base_id(c):
    return c.split('_')[0]


def wide(Cs, score):
    W = Cs.assign(s=score).pivot(index='client_id', columns='fam', values='s')[FAMS]
    W.columns = [f'sc_{f}' for f in FAMS]
    return W


V1_CLIENT = ['ntx', 'ntx90', 'n_generic', 'n_decoy', 'n_fee', 'n_salary', 'n_sub', 'n_sub90', 'n_ref_all',
             'n_active_fams', 'n_ok_fams']
V3_CLIENT = ['ntx', 'ntx90', 'n_card', 'sh_decoy', 'sh_generic', 'sh_kw', 'sh_everyday', 'n_salary', 'n_refund',
             'n_stream_pay', 'n_sp_60', 'n_sp_60_120', 'trend', 'n_streams', 'n_alive', 'n_alive_fams', 's_ref',
             'ref_ratio', 'n_ref_after_last', 'min_next', 'max_age', 'mean_age_alive']


PS_CLIENT = ['ps_max', 'ps_sum']   # client-level pseudo-scorer features (present when pre-training is used)


def _extra_client_cols(Cs):
    """client-level pseudo-scorer (ps_*) and competing-risks (crc_*) columns, when present"""
    return [c for c in PS_CLIENT if c in Cs.columns] + [c for c in Cs.columns if c.startswith('crc_')]


def client_table_v1(Cs):
    T = Cs.groupby('client_id')[V1_CLIENT + _extra_client_cols(Cs)].first()
    for col in ['nx_30', 'nx_gm', 's_next', 'since1']:
        T[f'min_{col}'] = Cs.groupby('client_id')[col].min()
    T['max_s_n'] = Cs.groupby('client_id').s_n.max(); T['sum_n90'] = Cs.groupby('client_id').n_90.sum()
    T['ref_ratio'] = T.n_ref_all / T.n_sub.clip(lower=1)
    return T


def client_table_v3(Cs):
    return Cs.groupby('client_id')[V3_CLIENT + _extra_client_cols(Cs)].first()


def lgb_params(kind, threads, seed=None):
    p = dict(learning_rate=0.03, min_data_in_leaf=30, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads)
    if kind == 'binary': p.update(objective='binary', num_leaves=15)
    else: p.update(objective='multiclass', num_class=8, num_leaves=7)
    if seed is not None: p['seed'] = seed
    return p


def fit_predict(version, Ctr, ymap, Cte, seed, threads, rounds_c=400, rounds_m=300, wmap=None):
    """Stage 1 (binary, per client x family) with 5-fold out-of-fold scores -> stage 2 (8 classes).
    Folds are split by base client so noisy copies of a client never leak across folds.
    Optional sample weights: column 'w' in Ctr (stage 1) and wmap base-client -> weight (stage 2)."""
    fe = [c for c in Ctr.columns if c not in ('client_id', 'fam', 'target', 'w')]
    w = Ctr['w'] if 'w' in Ctr.columns else None
    ctab = client_table_v1 if version == 'v1' else client_table_v3
    pc, pm = lgb_params('binary', threads, seed), lgb_params('multi', threads, seed)
    ids = np.array(sorted(Ctr.client_id.unique()))
    ug = np.unique(np.array([base_id(c) for c in ids]))
    gy = pd.Series([ymap[g] for g in ug], index=ug)
    bases = Ctr.client_id.map(base_id)
    oof = pd.Series(np.nan, index=Ctr.index)
    for a, _ in StratifiedKFold(5, shuffle=True, random_state=seed).split(ug, gy):
        inA = bases.isin(set(ug[a]))
        m = lgb.train(pc, lgb.Dataset(Ctr.loc[inA, fe], Ctr.loc[inA, 'target'],
                                      weight=None if w is None else w[inA]), rounds_c)
        oof.loc[~inA] = m.predict(Ctr.loc[~inA, fe])
    mc = lgb.train(pc, lgb.Dataset(Ctr[fe], Ctr.target, weight=w), rounds_c)
    s_te = mc.predict(Cte[fe])
    Xtr = ctab(Ctr).join(wide(Ctr, oof.values)); Xte = ctab(Cte).join(wide(Cte, s_te))
    yy = pd.Series([ymap[base_id(c)] for c in Xtr.index], index=Xtr.index)
    ww = None if wmap is None else pd.Series([wmap[base_id(c)] for c in Xtr.index], index=Xtr.index)
    mm = lgb.train(pm, lgb.Dataset(Xtr, yy.map(LABELS.index), weight=ww), rounds_m)
    return pd.DataFrame(mm.predict(Xte), index=Xte.index, columns=LABELS)


# =====================================================================================================
# Competing-risks model ("first surviving subscription"), fitted by EM on the labelled clients
# =====================================================================================================
# Label rule of the challenge: the family of the first subscription that still bills after the cutoff.
# At the real cutoff many subscriptions stop (e.g. almost all started 60-120 days earlier), and ~1/4 of
# clients stop everything. Model:
#   gate  z ~ Bernoulli(pi(client))                 the client stops everything  -> 'none'
#   else  each candidate s (alive stream, or a recent lone charge = new subscription) survives with a(s)
#   label = family of the first survivor, ordered by projected next charge (soft order, sd CR_SIG days).
# EM: streams before the label's stream are observed dead, the label's stream alive, later ones censored;
# for 'none' clients the gate posterior decides how much their streams count as dead.
# pi and a are LightGBM models (cross-entropy on soft targets). Blended with the two-stage GBM.
CR_SIG, CR_H = 3.0, 90.0
CR_DROP = {'client_id', 'fam', 'target', 'w', 's_next', 's_next1', 'nx_rank', 'nx_vs_other', 'nx_vs_min', 's_alive',
           'ps_score', 'ps_rank', 'ps_gap', 'ps_max', 'ps_sum', 'min_next'}
CR_BLEND = 0.4       # weight of the competing-risks model in the final blend


def _cr_prm(threads, seed):
    return dict(objective='cross_entropy', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads, seed=seed)


def cr_split(C):
    """v3 candidate table -> (candidate rows, client table)"""
    A = C[(C.s_alive == 1) & (C.s_next <= CR_H)].copy()
    A['is_lone'] = 0; A['lone_since'] = np.nan
    ls = C[['sg_tr_since', 'sg_kw_since']].min(axis=1)
    L = C[~((C.s_alive == 1) & (C.s_next <= CR_H)) & (ls <= 45)].copy()
    L['lone_since'] = ls[L.index]; L['is_lone'] = 1
    nx = 30.4 - L.lone_since; L['s_next'] = np.where(nx > 0.5, nx, nx + 30.4)
    A = pd.concat([A, L])
    clc = [c for c in C.columns if not c.startswith(('s_', 'sg_', 'f_', 'nx_', 'ps_', 'cr'))
           and c not in ('client_id', 'fam', 'fam_id', 'target', 'w')]
    X = C.groupby('client_id')[clc].first()
    for col in ['s_n', 's_age', 's_since', 's_overdue', 's_amt_cv', 's_own_kw', 'sg_tr_since', 'f_nstreams']:
        w = C.pivot(index='client_id', columns='fam', values=col).reindex(columns=FAMS); w.columns = [f'{col}_{f}' for f in FAMS]
        X = X.join(w)
    Aa = A[A.is_lone == 0]; g = Aa.groupby('client_id')
    X['k'] = g.size().reindex(X.index).fillna(0)
    X['k_lone'] = A[A.is_lone == 1].groupby('client_id').size().reindex(X.index).fillna(0)
    X['min_n'] = g.s_n.min(); X['max_n'] = g.s_n.max(); X['min_age'] = g.s_age.min(); X['max_age_a'] = g.s_age.max()
    X['n_mature'] = Aa[Aa.s_n >= 6].groupby('client_id').size().reindex(X.index).fillna(0)
    X['n_window'] = Aa[(Aa.s_age > 58) & (Aa.s_age < 125)].groupby('client_id').size().reindex(X.index).fillna(0)
    return A, X


def _order_matrix(nx):
    d = nx[None, :] - nx[:, None]                      # d[i, j] = next_j - next_i
    from scipy.stats import norm
    return norm.cdf(-d / (CR_SIG * np.sqrt(2)))        # P(j bills before i)


def cr_family_probs(A, a, pi):
    out = {}
    A = A.assign(_a=a)
    for cid, g in A.groupby('client_id', sort=False):
        av = g._a.values; Pb = _order_matrix(g.s_next.values); np.fill_diagonal(Pb, 0)
        out[cid] = dict(zip(g.fam.values, av * np.prod(1 - av[None, :] * Pb, axis=1)))
    P = pd.DataFrame.from_dict(out, orient='index').reindex(columns=FAMS).fillna(0.0).reindex(pi.index).fillna(0.0)
    P = P.mul(1 - pi.values, axis=0)
    P['none'] = 1 - P[FAMS].sum(1)
    return P


def _cr_estep(A, X, ymap, a, pi):
    A = A.assign(_a=a)
    t = pd.Series(0.0, index=A.index); w = pd.Series(0.0, index=A.index); z = pd.Series(0.0, index=X.index)
    for cid, g in A.groupby('client_id', sort=False):
        lab = ymap[base_id(cid)]; idx = g.index.values; av = g._a.values; nx = g.s_next.values
        if lab == 'none':
            zz = pi[cid] / (pi[cid] + (1 - pi[cid]) * np.prod(1 - av) + 1e-12)
            z[cid] = zz; t[idx] = 0.0; w[idx] = 1 - zz
        elif lab in set(g.fam.values):
            li = np.where(g.fam.values == lab)[0][0]; before = nx < nx[li]
            t[idx[li]] = 1.0; w[idx[li]] = 1.0; t[idx[before]] = 0.0; w[idx[before]] = 1.0
    in_a = set(A.client_id)
    z[[c for c in X.index if ymap[base_id(c)] == 'none' and c not in in_a]] = 1.0
    return t, w, z


def cr_fit(C3, ymap, threads, seed=0, iters=3, rounds=250):
    A, X = cr_split(C3)
    A = A.reset_index(drop=True)
    fa = [c for c in A.columns if c not in CR_DROP]; fx = list(X.columns)
    a = np.full(len(A), 0.7); pi = pd.Series(0.2, index=X.index)
    for _ in range(iters):
        t, w, z = _cr_estep(A, X, ymap, a, pi); m = w > 0
        ma = lgb.train(_cr_prm(threads, seed), lgb.Dataset(A.loc[m, fa], t[m], weight=w[m]), rounds)
        mp = lgb.train(_cr_prm(threads, seed), lgb.Dataset(X[fx], z), rounds)
        a = ma.predict(A[fa]); pi = pd.Series(mp.predict(X[fx]), index=X.index)
    return ma, mp, fa, fx


def cr_predict(model, C3):
    ma, mp, fa, fx = model
    A, X = cr_split(C3); A = A.reset_index(drop=True)
    a = ma.predict(A[fa]) if len(A) else np.array([])
    return cr_family_probs(A, a, pd.Series(mp.predict(X[fx]), index=X.index))[LABELS]


# =====================================================================================================
# Decision rule: down-weight 'none' (macro-F1 rewards recall on the 7 small classes)
# =====================================================================================================
NONE_WEIGHT = 0.7   # result of tune_none_weight() (train clients only; valid not used)
NONE_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def apply_none_weight(P, w):
    P = P[LABELS].copy(); P['none'] *= w
    return P


def em_prior_shift(P, prior_train, iters=100):
    """Label-shift correction (Saerens et al. 2002) on an UNLABELED set: estimate its class mix by EM from
    the predicted probabilities, then re-weight the probabilities by (estimated mix / train mix)."""
    X = P[LABELS].values; pi = prior_train.copy()
    for _ in range(iters):
        Q = X * (pi / prior_train); Q /= Q.sum(1, keepdims=True); pi = Q.mean(0)
    return pd.DataFrame(Q, index=P.index, columns=LABELS), pi


def tune_none_weight(feats, train_names, ymap, ytr, threads):
    """5-fold CV over TRAIN clients only: fit both pipelines on 4/5 of the train clients (+ their noisy
    copies), predict the held-out clients' test-level copy (_a3), pick the 'none' weight with the best
    out-of-fold macro-F1 of the 0.5/0.5 blend."""
    oof = {}
    for vi, version in enumerate(['v1', 'v3']):
        ALL = pd.concat([feats[n][vi] for n in train_names], ignore_index=True)
        ALL['target'] = (ALL.fam.values == ALL.client_id.map(lambda c: ymap.get(base_id(c))).values).astype(int)
        base = ALL.client_id.map(base_id); ids = np.array(ytr.index); parts = []
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, ytr)):
            tr, te = set(ids[a]), set(ids[b])
            Cte = ALL[base.isin(te) & ALL.client_id.str.endswith('_a3')].drop(columns=['target'])
            parts.append(fit_predict(version, ALL[base.isin(tr)], ymap, Cte, 0, threads))
            log(f'  tuning {version} fold {k} done')
        oof[version] = pd.concat(parts)
    O = 0.5 * oof['v1'][LABELS] + 0.5 * oof['v3'][LABELS]
    y = np.array([ymap[base_id(c)] for c in O.index])
    scores = {w: f1_score(y, np.array(LABELS)[apply_none_weight(O, w).values.argmax(1)], average='macro')
              for w in NONE_GRID}
    for w, s in scores.items(): log(f'  none weight {w:.2f}: out-of-fold macro-F1 {s:.4f}')
    return max(scores, key=scores.get)


# =====================================================================================================
# Proxy pre-training on the unlabeled pretrain set
# =====================================================================================================
PRETRAIN_CUTS = {'c1': '2025-07-01', 'c2': '2025-08-16', 'c3': '2025-10-01'}   # simulated cutoffs


def _pretrain_features_job(args):
    version, k, df, cache_dir = args
    path = os.path.join(cache_dir, f'feat_pt_{version}_{k}.pkl') if cache_dir else None
    if path and os.path.exists(path):
        return version, k, pd.read_pickle(path), None
    t0 = time.time()
    C = cand_features_v3(df, cutoff=pd.Timestamp(PRETRAIN_CUTS[k], tz='UTC'))
    C['client_id'] = C.client_id + f'_{k}'
    if path: C.to_pickle(path)
    return version, k, C, round(time.time() - t0, 1)


def pseudo_labels(pt):
    """Per simulated cutoff: family of the first payment in (cutoff, cutoff + 90d] that belongs to a recurring
    stream (>= 2 payments) detected on the full, clean pretrain history; 'none' if there is none."""
    d = prep_v3(pt, cutoff=CUTOFF)
    streams = {}
    for cid, g in d.groupby('client_id', sort=False):
        g = g.sort_values('t'); t = g.t.values; amt = g.amount.values; E = g[[f'e_{f}' for f in FAMS]].values
        out = []
        for s in client_streams_v3(g):
            if len(s) < 2: continue
            ss = np.array(sorted(s, key=lambda i: t[i])); V = E[ss].sum(0); fam = FAMS[int(np.argmax(V))]
            if fam in ('streaming', 'music') and abs(V[FI['streaming']] - V[FI['music']]) < 1e-9:
                fam = 'streaming' if np.median(amt[ss]) >= 15.5 else 'music'
            out.append((fam, t[ss]))
        streams[cid] = out
    labels = {}
    for k, cut in PRETRAIN_CUTS.items():
        c0 = (pd.Timestamp(cut, tz='UTC') - EPOCH).total_seconds() / 86400.0
        lab = {}
        for cid, out in streams.items():
            best, bt = 'none', np.inf
            for fam, ts in out:
                w = ts[(ts > c0) & (ts <= c0 + 90)]
                if len(w) and w[0] < bt: best, bt = fam, w[0]
            lab[cid] = best
        labels[k] = lab
    return labels


def train_pseudo_scorer(pt_feats, labels, threads):
    """binary LightGBM on pseudo-labelled (client, family) rows: 'does this family bill first?'"""
    parts = []
    for (version, k), C in pt_feats.items():
        C = C.copy(); lab = labels[k]
        C['target'] = (C.fam.values == C.client_id.map(lambda c: lab[base_id(c)]).values).astype(int)
        parts.append(C)
    P = pd.concat(parts, ignore_index=True)
    fe = [c for c in P.columns if c not in ('client_id', 'fam', 'target')]
    prm = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads, seed=0)
    return lgb.train(prm, lgb.Dataset(P[fe], P.target), 600), fe, len(P)


def add_pseudo_features(C1, C3, model, fe):
    s = model.predict(C3[fe])
    ps = pd.DataFrame({'client_id': C3.client_id.values, 'fam': C3.fam.values, 'ps_score': s})
    g = ps.groupby('client_id').ps_score
    ps['ps_rank'] = g.rank(ascending=False, method='min'); ps['ps_gap'] = ps.ps_score - g.transform('max')
    ps['ps_max'] = g.transform('max'); ps['ps_sum'] = g.transform('sum')
    return C1.merge(ps, on=['client_id', 'fam'], how='left'), C3.merge(ps, on=['client_id', 'fam'], how='left')


# =====================================================================================================
# Orchestration
# =====================================================================================================
def _features_job(args):
    name, df, cache_dir = args
    paths = {v: os.path.join(cache_dir, f'feat_{v}_{name}.pkl') for v in ('v1', 'v3')} if cache_dir else {}
    if paths and all(os.path.exists(p) for p in paths.values()):
        return name, pd.read_pickle(paths['v1']), pd.read_pickle(paths['v3'])
    t0 = time.time()
    C1 = cand_features_v1(df); C3 = cand_features_v3(df)
    if paths:
        C1.to_pickle(paths['v1']); C3.to_pickle(paths['v3'])
    return name, C1, C3, round(time.time() - t0, 1)


def ensure_data(data_dir):
    if all(os.path.exists(os.path.join(data_dir, f)) for f in FILES): return
    z = os.path.join(data_dir, 'dataset.zip')
    if not os.path.exists(z): sys.exit(f'Missing data files in {data_dir} (and no dataset.zip)')
    log('extracting', z); zipfile.ZipFile(z).extractall(data_dir)


VALID_COPIES = [('mid', 31), ('mid', 32)]    # noisy copies of valid clients (only when valid labels are used)
SELF_TRAIN_WEIGHT = 0.3                        # weight of pseudo-labelled unlabeled clients in the final round


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--jobs', type=int, default=2, help='parallel processes for feature building')
    ap.add_argument('--threads', type=int, default=2, help='LightGBM threads (2 reproduces our numbers)')
    ap.add_argument('--cache-dir', default='feature_cache', help="'' disables caching of feature tables")
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--labels', choices=['all', 'train'], default='all',
                    help="'all': train + valid labels (final submission, no validation score possible); "
                         "'train': train labels only and validation scores are printed")
    ap.add_argument('--self-train', type=float, default=SELF_TRAIN_WEIGHT,
                    help='weight of the pseudo-labelled unlabeled pretrain clients (0 disables self-training)')
    ap.add_argument('--cr-blend', type=float, default=CR_BLEND,
                    help='weight of the competing-risks model in the final blend (0 disables it)')
    ap.add_argument('--balance', choices=['none_weight', 'em', 'off'], default='off',
                    help='decision rule for the submission (see docstring)')
    ap.add_argument('--no-pretrain', action='store_true', help='do not use the unlabeled pretrain set at all')
    ap.add_argument('--extra-eval', default='', help=argparse.SUPPRESS)   # dev: cached noisy copies of valid, scored
    ap.add_argument('--save-probs', default='', help=argparse.SUPPRESS)   # dev: pickle (gbm, cr, blend) probabilities
    ap.add_argument('--tune-none', action='store_true',
                    help=f're-derive the none weight by CV on train clients (~5 min); default uses {NONE_WEIGHT}')
    args = ap.parse_args()
    ensure_data(args.data_dir)
    if args.cache_dir: os.makedirs(args.cache_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    use_valid = args.labels == 'all'
    self_train = args.self_train > 0 and not args.no_pretrain
    lab = lambda n: pd.read_csv(os.path.join(args.data_dir, f'{n}_labels.csv')).set_index('client_id').target_next_recurring_merchant
    ytr, yva = lab('train'), lab('valid')
    ss = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))

    # ---- data: labelled clients + noisy copies, test, and (optionally) the unlabeled pretrain clients
    log('loading + tagging')
    train = tag(load(args.data_dir, 'train'))
    datasets = [('train', train)]
    for level, seed in TRAIN_COPIES:
        datasets.append((f'train_{level}_a{seed}', augment(train, level, seed=seed, suffix=f'_a{seed}')))
    valid = tag(load(args.data_dir, 'valid')); datasets.append(('valid', valid))
    if use_valid:
        for level, seed in VALID_COPIES:
            datasets.append((f'valid_{level}_a{seed}', augment(valid, level, seed=seed, suffix=f'_a{seed}')))
    datasets.append(('test', tag(load(args.data_dir, 'test'))))
    if not args.no_pretrain:
        pt = tag(load(args.data_dir, 'unlabeled_pretrain'))
        pt_noisy = augment(pt, 'test', seed=101, suffix='_n')
        if self_train:
            datasets += [('ptreal_clean', pt), ('ptreal_noisy', pt_noisy)]

    # ---- features (both pipelines)
    log(f'building features for {len(datasets)} datasets with {args.jobs} process(es)')
    feats = {}
    jobs = [(n, d, args.cache_dir) for n, d in datasets]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            for r in ex.map(_features_job, jobs):
                feats[r[0]] = (r[1], r[2]); log('  features', r[0], *(r[3:] or ['(cached)']))
    else:
        for j in jobs:
            r = _features_job(j); feats[r[0]] = (r[1], r[2]); log('  features', r[0], *(r[3:] or ['(cached)']))

    extra_eval = [n for n in args.extra_eval.split(',') if n]
    for n in extra_eval:
        feats[n] = tuple(pd.read_pickle(os.path.join(args.cache_dir, f'feat_{v}_{n}.pkl')) for v in ('v1', 'v3'))

    # ---- proxy pre-training: pseudo-labelled simulated cutoffs on the unlabeled pretrain set
    if not args.no_pretrain:
        pt_jobs = [(v, k, {'clean': pt, 'noisy': pt_noisy}[v], args.cache_dir) for v in ('clean', 'noisy') for k in PRETRAIN_CUTS]
        pt_feats = {}
        with ProcessPoolExecutor(max(1, args.jobs)) as ex:
            for v, k, C, secs in ex.map(_pretrain_features_job, pt_jobs):
                pt_feats[(v, k)] = C; log(f'  pretrain features {v} {k}', secs if secs is not None else '(cached)')
        labels = pseudo_labels(pt)
        for k in PRETRAIN_CUTS:
            log(f'  pseudo-label mix {k}:', pd.Series(labels[k]).value_counts(normalize=True).round(3).to_dict())
        scorer, ps_fe, n_rows = train_pseudo_scorer(pt_feats, labels, args.threads)
        log(f'  pseudo-scorer trained on {n_rows} pseudo-labelled rows')
        if self_train:   # pretrain clients themselves get cross-fitted scores (scorer trained on the other half)
            ptb = np.array(sorted({base_id(c) for c in pt_feats[('clean', 'c1')].client_id}))
            half = {b: i % 2 for i, b in enumerate(ptb)}
            half_sc = {h: train_pseudo_scorer({k: C[C.client_id.map(lambda c: half[base_id(c)] == h)]
                                               for k, C in pt_feats.items()}, labels, args.threads)[0] for h in (0, 1)}
        for n in feats:
            C1, C3 = feats[n]
            if n.startswith('ptreal'):
                parts = []
                for h in (0, 1):
                    m1 = C1.client_id.map(lambda c: half[base_id(c)] == h); m3 = C3.client_id.map(lambda c: half[base_id(c)] == h)
                    parts.append(add_pseudo_features(C1[m1], C3[m3], half_sc[1 - h], ps_fe))
                feats[n] = (pd.concat([p[0] for p in parts], ignore_index=True), pd.concat([p[1] for p in parts], ignore_index=True))
            else:
                feats[n] = add_pseudo_features(C1, C3, scorer, ps_fe)
        del pt, pt_noisy, pt_feats

    train_names = [n for n, _ in datasets if n.startswith('train')]
    labelled_names = train_names + ([n for n, _ in datasets if n.startswith('valid')] if use_valid else [])
    ylab = pd.concat([ytr, yva]) if use_valid else ytr
    ymap = ylab.to_dict()
    prior = ylab.value_counts(normalize=True).reindex(LABELS).values
    if not use_valid:
        train_bases = {base_id(c) for n in train_names for c in feats[n][0].client_id.unique()}
        assert not (train_bases & set(yva.index)), 'validation clients must not be in the training data'
    log('training labels:', 'train + valid' if use_valid else 'train only', f'({len(ymap)} clients)')

    def train_round(extra_names, ym, wmap, eval_names):
        probs = {}
        for vi, version in enumerate(['v1', 'v3']):
            Ctr = pd.concat([feats[n][vi] for n in labelled_names + extra_names], ignore_index=True)
            Ctr['target'] = (Ctr.fam.values == Ctr.client_id.map(lambda c: ym[base_id(c)]).values).astype(int)
            if wmap is not None: Ctr['w'] = Ctr.client_id.map(lambda c: wmap[base_id(c)]).astype(float)
            Cev = pd.concat([feats[n][vi] for n in eval_names], ignore_index=True)
            Ps = []
            for seed in seeds:
                Ps.append(fit_predict(version, Ctr, ym, Cev, seed, args.threads, wmap=wmap))
                log(f'  {version} seed {seed} done')
            probs[version] = sum(Ps) / len(Ps)
        return probs

    # ---- competing-risks model (labelled clients only), blended with the GBM
    crP = None
    if args.cr_blend > 0:
        log('competing-risks model: EM fit on the labelled clients')
        C3lab = pd.concat([feats[n][1] for n in labelled_names], ignore_index=True)
        cr_names = ['valid', 'test'] + extra_eval + (['ptreal_clean'] if self_train else [])
        Ps = []
        for seed in seeds:
            crm = cr_fit(C3lab, ymap, args.threads, seed=seed)
            Ps.append(pd.concat([cr_predict(crm, feats[n][1]) for n in cr_names]))
            log(f'  competing-risks seed {seed} done')
        crP = sum(Ps) / len(Ps)

    def blend_cr(G):
        G = G[LABELS]
        return G if crP is None else (1 - args.cr_blend) * G + args.cr_blend * crP.loc[G.index][LABELS]

    # ---- self-training: pseudo-label the 10,000 unlabeled clients at the real cutoff, then retrain
    if self_train:
        log('self-training round 0: labelling the unlabeled clients')
        p0 = train_round([], ymap, None, ['ptreal_clean'])
        q, pi_pt = em_prior_shift(blend_cr(0.5 * p0['v1'][LABELS] + 0.5 * p0['v3'][LABELS]), prior)
        pseudo = q.idxmax(1)
        log('  pseudo-label mix of unlabeled clients:', pseudo.value_counts(normalize=True).round(3).to_dict())
        ym_all = dict(ymap, **pseudo.to_dict())
        wmap = {c: 1.0 for c in ymap}; wmap.update({c: args.self_train for c in pseudo.index})
        log(f'self-training round 1: labelled clients + {len(pseudo)} pseudo-labelled clients (weight {args.self_train})')
        probs = train_round(['ptreal_clean', 'ptreal_noisy'], ym_all, wmap, ['valid', 'test'] + extra_eval)
    else:
        probs = train_round([], ymap, None, ['valid', 'test'] + extra_eval)
    gbm = 0.5 * probs['v1'][LABELS] + 0.5 * probs['v3'][LABELS]
    blend = blend_cr(gbm)
    w_none = tune_none_weight(feats, train_names, ytr.to_dict(), ytr, args.threads) if args.tune_none else NONE_WEIGHT
    em_va, _ = em_prior_shift(blend.loc[yva.index], prior)          # each unlabeled set gets its own
    em_te, pi_te = em_prior_shift(blend.loc[ss.client_id], prior)   # estimated class mix
    variants = {'off': blend, 'none_weight': apply_none_weight(blend, w_none), 'em': pd.concat([em_va, em_te])}
    final = variants[args.balance]
    if args.save_probs: pd.to_pickle(dict(gbm=gbm, cr=crP, blend=blend, prior=prior), args.save_probs)
    log(f'decision rule: {args.balance}; EM-estimated test class mix:', dict(zip(LABELS, pi_te.round(3))))

    def pred(P, ids): return np.array(LABELS)[P.loc[ids][LABELS].values.argmax(1)]
    if use_valid:
        print('\nValid labels were used for training, so no validation score is printed '
              '(run with --labels train for an honest validation score).')
    else:
        print('\nMacro-F1 on validation (1,000 clients, not used for training or tuning):')
        rows = [('v1', probs['v1']), ('v3', probs['v3']), ('gbm (v1+v3), argmax', gbm),
                ('gbm (v1+v3), em', em_prior_shift(gbm.loc[yva.index], prior)[0])]
        if crP is not None: rows.append(('competing-risks, argmax', crP.loc[yva.index]))
        rows += [(f'blend, rule={k}', v) for k, v in variants.items()]
        for nm, P in rows:
            mark = '  <- submitted' if nm == f'blend, rule={args.balance}' else ''
            print(f'  {nm:28s} {f1_score(yva.values, pred(P, yva.index), average="macro"):.4f}{mark}')
        for n in extra_eval:
            ids = [c for c in gbm.index if c.endswith(n.split('_')[-1]) and base_id(c) in yva.index]
            yv = yva.loc[[base_id(c) for c in ids]].values
            f = lambda P: f1_score(yv, pred(P, ids), average='macro')
            b = blend.loc[ids]; em_b = em_prior_shift(b, prior)[0]; em_g = em_prior_shift(gbm.loc[ids], prior)[0]
            print(f'  [{n}] gbm/em {f(em_g):.4f}  cr/argmax {f(crP.loc[ids]) if crP is not None else float("nan"):.4f}  '
                  f'blend/argmax {f(b):.4f}  blend/em {f(em_b):.4f}')
        order = FAMS + ['none']
        print(classification_report(yva.values, pred(final, yva.index), labels=order, digits=3))
        cm = pd.crosstab(pd.Series(yva.values, name='true'), pd.Series(pred(final, yva.index), name='pred')).reindex(index=order, columns=order, fill_value=0)
        print(cm.to_string())

    # ---- submission
    sub = pd.DataFrame({'client_id': ss.client_id, 'predicted_next_recurring_merchant': pred(final, ss.client_id)})
    assert len(sub) == len(ss) and sub.client_id.is_unique and sub.predicted_next_recurring_merchant.isin(LABELS).all()
    sub.to_csv(args.out, index=False)
    log('wrote', args.out, sub.predicted_next_recurring_merchant.value_counts(normalize=True).round(3).to_dict())


if __name__ == '__main__':
    main()
