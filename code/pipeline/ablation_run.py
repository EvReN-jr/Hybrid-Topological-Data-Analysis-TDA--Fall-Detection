"""Faithful General/LOSO replication for feature-block ablation.
Protocol matches uhem_big_optuna_v13.py: global GridSearchCV once on
balanced-undersampled data (cv=3, F2 scorer), then per LOSO fold 15 reps
(undersample seed=rep, fit, PR-curve threshold, predict), per-subject mean,
then cross-subject mean +/- 95% t-CI.
Usage: python ablation_run.py <LogReg|SVM>
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sqlite3, json, sys, time, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, recall_score
from scipy import stats as scipy_stats
from joblib import Parallel, delayed

N_REP = 15
f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
DB = "/home/evo/YL_TEZ/tda_proje/Results_V15.db"

MODELS = {
    "LogReg": dict(
        est=LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, tol=1e-3),
        params={"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"]}),
    "SVM": dict(
        est=SVC(class_weight="balanced", probability=True, cache_size=2000),
        params={"clf__C": [0.1, 1, 10, 100], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale"]}),
}
BLOCKS = {"Full": list(range(24)), "H0": list(range(0, 10)),
          "H1": list(range(10, 20)), "H0+H1": list(range(0, 20)),
          "Signal": list(range(20, 24))}
DATASETS = ["MobiFall", "SisFall", "FAD_40Hz"]


def undersample(X, y, seed):
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if len(i1) == 0:
        return X, y
    n0 = min(len(i0), len(i1) * 3)
    if n0 < len(i0):
        c0 = rng.choice(i0, n0, replace=False)
        return np.concatenate([X[c0], X[i1]]), np.concatenate([y[c0], y[i1]])
    return X, y


def best_thr(model, Xv, yv):
    try:
        pr = model.predict_proba(Xv)[:, 1]
        p, r, t = precision_recall_curve(yv, pr)
        f2 = np.divide(5 * p * r, 4 * p + r, out=np.zeros_like(p), where=(4 * p + r) != 0)
        return float(t[np.argmax(f2)]) if len(t) > 0 else 0.5
    except Exception:
        return 0.5


def mean_ci(vals, conf=0.95):
    a = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(a) < 2:
        return (float(a.mean()) if len(a) else 0.0), 0.0
    m = float(a.mean())
    se = scipy_stats.sem(a)
    return m, float(se * scipy_stats.t.ppf((1 + conf) / 2.0, len(a) - 1))


def eval_fold(Xtr, ytr, Xte, yte, pipe):
    f2s, recs = [], []
    for rep in range(N_REP):
        Xb, yb = undersample(Xtr, ytr, rep)
        pipe.fit(Xb, yb)
        thr = best_thr(pipe, Xb, yb)
        pred = (pipe.predict_proba(Xte)[:, 1] >= thr).astype(int)
        f2s.append(fbeta_score(yte, pred, beta=2, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
    return float(np.mean(f2s)), float(np.mean(recs))


def main(model_name):
    con = sqlite3.connect(DB)
    out = {}
    for ds in DATASETS:
        P = pd.read_sql(f"SELECT * FROM features_{ds}_server", con)
        Xall = P[[f"feat_{i}" for i in range(24)]].values
        yall = P["label"].astype(int).values
        gall = P["subject"].values
        logo = LeaveOneGroupOut()
        mc = MODELS[model_name]
        for bname, cols in BLOCKS.items():
            t0 = time.time()
            Xb = Xall[:, cols]
            X0, y0 = undersample(Xb, yall, 0)
            pipe = Pipeline([("sc", StandardScaler()), ("clf", clone(mc["est"]))])
            gs = GridSearchCV(pipe, mc["params"], cv=3, scoring=f2_scorer, n_jobs=-1)
            gs.fit(X0, y0)
            best = gs.best_estimator_
            folds = list(logo.split(Xb, yall, gall))
            res = Parallel(n_jobs=-1)(
                delayed(eval_fold)(Xb[tr], yall[tr], Xb[te], yall[te], clone(best))
                for tr, te in folds)
            sf2 = [r[0] for r in res]
            srec = [r[1] for r in res]
            m, h = mean_ci(sf2)
            rm, rh = mean_ci(srec)
            key = f"{ds}|{model_name}|{bname}"
            out[key] = dict(f2=m, f2_ci=h, rec=rm, rec_ci=rh,
                            best_params=str(gs.best_params_), n_subj=len(sf2))
            print(f"{key:32s} F2={m:.4f}+-{h:.4f}  Rec={rm:.4f}  "
                  f"{gs.best_params_}  ({time.time()-t0:.0f}s)", flush=True)
            # incremental save
            fn = f"/home/evo/YL_TEZ/tda_proje/ablation_{model_name}.json"
            json.dump(out, open(fn, "w"), indent=2)
    print(f"DONE {model_name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LogReg")
