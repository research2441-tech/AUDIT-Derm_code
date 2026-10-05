"""Build cohorts, lesion-grouped splits, the frozen source classifier, and per-image scores.

Outputs data/cases.csv with one row per image: cohort, condition, label, split, logit score.
"""
import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20261003)

COHORT = {163: 'MSKCC', 212: 'HAM', 249: 'BCN', 251: 'HIBA', 406: 'PAD', 425: 'MILK'}


def condition(row):
    t = row.image_type
    if row.cohort == 'PAD':
        return 'smartphone'
    if t == 'dermoscopic':
        return 'dermoscopic'
    if t == 'clinical: overview':
        return 'overview'
    return 'closeup'


def main():
    df = pd.read_csv('index.csv')
    emb = np.load('emb.npy')
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    df['cohort'] = df.coll.map(COHORT)
    df['condition'] = df.apply(condition, axis=1)
    df['y'] = (df.dx1 == 'Malignant').astype(int)
    # exclusions: indeterminate melanocytic lesions and missing diagnosis; actinic keratosis kept as non-malignant
    excl = (df.dx1.isna()) | ((df.dx1 == 'Indeterminate') & (df.dx3 != 'Solar or actinic keratosis'))
    df['excluded'] = excl
    # lesion identifier for grouping (MSKCC lacks lesion ids -> each image its own group)
    df['group'] = np.where(df.lesion_id.notna(), df.lesion_id, df.id)
    # MILK lesions that also appear in HAM are removed from the development data (leakage control)
    milk_les = set(df.loc[df.cohort == 'MILK', 'lesion_id'])
    df['overlap'] = (df.cohort == 'HAM') & df.lesion_id.isin(milk_les)

    # ---------- paired lesions (same lesion, clinical + dermoscopic) ----------
    df['split'] = 'pool'
    paired = {}
    for site in ['HIBA', 'MILK']:
        s = df[(df.cohort == site) & ~df.excluded]
        g = s.groupby('group').condition.apply(set)
        les = sorted([k for k, v in g.items() if 'dermoscopic' in v and len(v) > 1])
        RNG.shuffle(les)
        half = len(les) // 2
        fit = set(les[:half])
        df.loc[(df.cohort == site) & df.group.isin(fit), 'split'] = 'pairfit'
        paired[site] = dict(n_paired=len(les), n_fit=half)

    # ---------- development data: HAM (minus overlap) + BCN, lesion-grouped 60/20/20 ----------
    dev = (df.cohort.isin(['HAM', 'BCN'])) & ~df.excluded & ~df.overlap
    groups = df.loc[dev, 'group'].unique()
    RNG.shuffle(groups)
    n = len(groups)
    tr, ca = set(groups[:int(.6 * n)]), set(groups[int(.6 * n):int(.8 * n)])
    df.loc[dev, 'split'] = np.where(df.loc[dev, 'group'].isin(tr), 'train',
                                    np.where(df.loc[dev, 'group'].isin(ca), 'cal', 'test'))
    df.loc[df.overlap, 'split'] = 'dropped'
    df.loc[df.excluded, 'split'] = 'excluded'

    # ---------- frozen source classifier: L2-regularised logistic head on CLIP embeddings ----------
    trm = (df.split == 'train').values
    Xtr, ytr, gtr = emb[trm], df.y.values[trm], df.group.values[trm]
    best = None
    for C in [0.3, 1, 3, 10, 30]:
        aucs = []
        for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
            m = LogisticRegression(C=C, max_iter=3000).fit(Xtr[a], ytr[a])
            aucs.append(roc_auc_score(ytr[b], m.decision_function(Xtr[b])))
        if best is None or np.mean(aucs) > best[1]:
            best = (C, np.mean(aucs))
    clf = LogisticRegression(C=best[0], max_iter=5000).fit(Xtr, ytr)
    df['u'] = clf.decision_function(emb)  # raw logit score of the frozen head
    np.save('head_w.npy', np.r_[clf.coef_.ravel(), clf.intercept_])
    info = dict(C=best[0], cv_auc=best[1], paired=paired,
                n_by_split=df.split.value_counts().to_dict())
    for k in ['test', 'cal']:
        m = df.split == k
        info['auc_' + k] = roc_auc_score(df.y[m], df.u[m])
    for (c, cond), s in df[(df.split == 'pool')].groupby(['cohort', 'condition']):
        if s.y.nunique() > 1:
            info[f'auc_{c}_{cond}'] = (roc_auc_score(s.y, s.u), len(s), s.y.mean())
    print(json.dumps(info, indent=1, default=str))
    json.dump(info, open('prepare_info.json', 'w'), indent=1, default=str)
    df.drop(columns=['off', 'len']).to_csv('cases.csv', index=False)


if __name__ == '__main__':
    main()
