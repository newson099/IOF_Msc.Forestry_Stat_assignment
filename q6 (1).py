"""
Q6. Perceived change in (a) condition of biodiversity and (b) number of
    wild animals, BEFORE vs AFTER buffer zone (BZ) implementation.
    Tested with the non-parametric Wilcoxon Signed-Rank test (used instead
    of a paired t-test because the responses are ordinal: 1=Low, 2=Average,
    3=High -- not continuous/interval data).

"""

import os
import pandas as pd
from scipy.stats import wilcoxon

DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q6_results.xlsx")

ALPHA = 0.05  # standard 5% significance level (question does not specify one)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

label_map = {1: "Low", 2: "Average", 3: "High"}


def run_wilcoxon(data, before_col, after_col, name, alpha=ALPHA):
    before = data[before_col]
    after = data[after_col]

    freq_table = pd.DataFrame({
        "Before": before.map(label_map).value_counts().reindex(["Low", "Average", "High"]),
        "After": after.map(label_map).value_counts().reindex(["Low", "Average", "High"]),
    })

    diff = after - before
    n_increase = (diff > 0).sum()
    n_decrease = (diff < 0).sum()
    n_tie = (diff == 0).sum()

    stat, p_value = wilcoxon(before, after)

    print(f"--- {name}: Before vs After ---")
    print(freq_table, "\n")
    print(f"Respondents reporting increase: {n_increase}, decrease: {n_decrease}, no change: {n_tie}")
    print(f"Wilcoxon signed-rank test: W = {stat:.3f}, p-value = {p_value:.4f}")
    if p_value < alpha:
        print(f"-> Reject H0 at {int(alpha*100)}% level: significant change in perception.\n")
    else:
        print(f"-> Fail to reject H0 at {int(alpha*100)}% level: no significant change.\n")

    result = pd.DataFrame([{
        "N": len(data),
        "N_Increase": n_increase,
        "N_Decrease": n_decrease,
        "N_Tie": n_tie,
        "W_statistic": round(stat, 3),
        "p_value": round(p_value, 4),
        "Alpha": alpha,
        "Significant": p_value < alpha,
    }])
    return freq_table, result


# Biodiversity condition: Before_BD vs After_BD 
print("=" * 65)
print("(a) PERCEIVED CONDITION OF BIODIVERSITY")
print("=" * 65)
bd_freq, bd_result = run_wilcoxon(df, "Before_BD", "After_BD", "Biodiversity Condition")

# Number of wild animals: Before_WL vs After_WL 
print("=" * 65)
print("(b) PERCEIVED NUMBER OF WILD ANIMALS")
print("=" * 65)
wl_freq, wl_result = run_wilcoxon(df, "Before_WL", "After_WL", "Wildlife Numbers")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    bd_freq.to_excel(writer, sheet_name="Biodiversity_Frequencies")
    bd_result.to_excel(writer, sheet_name="Biodiversity_Wilcoxon", index=False)
    wl_freq.to_excel(writer, sheet_name="Wildlife_Frequencies")
    wl_result.to_excel(writer, sheet_name="Wildlife_Wilcoxon", index=False)

print(f"All Q6 results saved to: {OUTPUT_FILE}")
