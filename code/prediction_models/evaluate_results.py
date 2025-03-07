# %%
import os, pickle
import numpy as np
from scipy.stats import binomtest, wilcoxon, shapiro, ttest_rel
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# %%
with open(os.path.join(BASE, 'large_files/20250211-123659-reg.pkl'),
          'rb') as fi:
    results_wo_boxes = pickle.load(fi)['metrics']
with open(os.path.join(BASE, 'large_files/20250202-102800-reg.pkl'),
          'rb') as fi:
    results_boxes = pickle.load(fi)['metrics']
# %%
r2_wo_boxes = np.array([r['best_metric'] for r in results_wo_boxes])
r2_boxes = np.array([r['best_metric'] for r in results_boxes])
diff = r2_boxes - r2_wo_boxes
print(diff)
print(binomtest(sum(diff>0), 10))

stat, p_value = wilcoxon(r2_boxes, r2_wo_boxes, alternative='greater')
print(stat)
print(p_value)

shapiro_stat, shapiro_p = shapiro(diff)
print(f"Shapiro-Wilk test statistic: {shapiro_stat}")
print(f"Shapiro-Wilk p-value: {shapiro_p}")

t_stat, p_value = ttest_rel(r2_boxes, r2_wo_boxes, alternative='greater')
print(f"Paired t-test statistic: {t_stat}")
print(f"P-value: {p_value}")
