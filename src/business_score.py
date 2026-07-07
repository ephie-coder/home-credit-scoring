import numpy as np
from sklearn.metrics import confusion_matrix

# Coûts métier (contexte crédit bancaire)
COST_FALSE_NEGATIVE = 5000   # prêt accordé à quelqu'un qui fait défaut
COST_FALSE_POSITIVE = 500    # bon client refusé à tort

def business_cost(y_true, y_pred, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * cost_fn) + (fp * cost_fp)

def business_score_normalized(y_true, y_pred, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    cost = business_cost(y_true, y_pred, cost_fn, cost_fp)
    max_cost = sum(y_true) * cost_fn + sum(1 - y_true) * cost_fp
    return round(1 - cost / max_cost, 4)

def find_optimal_threshold(y_true, y_proba, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE):
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_cost = 0.5, float('inf')
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        c = business_cost(y_true, y_pred, cost_fn, cost_fp)
        if c < best_cost:
            best_cost = c
            best_t = t
    return best_t, best_cost