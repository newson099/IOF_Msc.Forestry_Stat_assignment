"""
Q4. Is there any change of average no. of days per week to go to the park before and after declaration of BZ?
Use paired t test. Interpret your result statistically.
"""

import os
import pandas as pd
from scipy import stats

DATA_PATH = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\spss.csv"
OUTPUT_DIR = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q4_results.xlsx")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

before = df["Befor_BZ_days"]
after = df["After_BZ_days"]

# Descriptive statistics
desc = pd.DataFrame({
    "Before_BZ": before.describe(),
    "After_BZ": after.describe(),
}).round(2)
print("Park visits per week - Before vs After BZ declaration")
print(desc, "\n")

# Paired t-test
# H0: mean(Before) = mean(After)   H1: mean(Before) != mean(After)
t_stat, p_value = stats.ttest_rel(before, after)
mean_diff = (before - after).mean()
sd_diff = (before - after).std()

print(f"Mean difference (Before - After): {mean_diff:.3f}")
print(f"Std. Dev of differences: {sd_diff:.3f}")
print(f"Paired t-test: t = {t_stat:.3f}, df = {len(df)-1}, p = {p_value:.4f}")
if p_value < 0.05:
    print("-> Significant difference at 5% level: visit frequency changed after BZ declaration.\n")
else:
    print("-> No significant difference at 5% level.\n")

result_table = pd.DataFrame([{
    "N": len(df),
    "Mean_Before": round(before.mean(), 3),
    "Mean_After": round(after.mean(), 3),
    "Mean_Difference": round(mean_diff, 3),
    "SD_Difference": round(sd_diff, 3),
    "t_stat": round(t_stat, 3),
    "df": len(df) - 1,
    "p_value": round(p_value, 4),
    "Significant_at_5pct": p_value < 0.05,
}])
print(result_table.to_string(index=False))

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    desc.to_excel(writer, sheet_name="Descriptives")
    result_table.to_excel(writer, sheet_name="Paired_t_test", index=False)

print(f"\nAll Q4 results saved to: {OUTPUT_FILE}")
