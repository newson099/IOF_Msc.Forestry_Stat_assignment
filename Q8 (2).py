"""
Q8. Total loss per household is extremely non-normal, so non-parametric
    rank-based tests are used instead of t-test/ANOVA:
      - Mann-Whitney U test : Total loss vs District (Bara vs Parsa)
      - Kruskal-Wallis H test : Total loss vs Economic Class (Poor/Middle/Rich)
"""

import os
import pandas as pd
from scipy.stats import shapiro, mannwhitneyu, kruskal

DATA_PATH = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\spss.csv"
OUTPUT_DIR = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\8"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q8_results.xlsx")


df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

df["Total_loss"] = (
    df["Rice_price"] + df["Maize_price"] + df["Veg_price"] + df["Fruit_price"]
    + df["Cow.Price"] + df["Buf.Price"] + df["Gt.Price"]
)

df["District_lbl"] = df["District"].map({1: "Bara", 2: "Parsa"})
df["Eco_class_lbl"] = df["Eco_class"].map({1: "Poor", 2: "Middle", 3: "Rich"})

# Normality check (Shapiro-Wilk) on Total_loss, overall and by group ----
print("Shapiro-Wilk normality test on Total_loss (H0: data is normally distributed)\n")

stat, p = shapiro(df["Total_loss"])
print(f"Overall sample: W = {stat:.4f}, p = {p:.6f}  ->  {'Normal' if p > 0.05 else 'NOT normal'}")

normality_rows = [{"Group": "Overall", "W_statistic": round(stat, 4), "p_value": round(p, 6), "Normal": p > 0.05}]

for grp, sub in df.groupby("District_lbl"):
    stat, p = shapiro(sub["Total_loss"])
    print(f"District = {grp}: W = {stat:.4f}, p = {p:.6f}  ->  {'Normal' if p > 0.05 else 'NOT normal'}")
    normality_rows.append({"Group": f"District: {grp}", "W_statistic": round(stat, 4), "p_value": round(p, 6), "Normal": p > 0.05})

for grp, sub in df.groupby("Eco_class_lbl"):
    stat, p = shapiro(sub["Total_loss"])
    print(f"Economic Class = {grp}: W = {stat:.4f}, p = {p:.6f}  ->  {'Normal' if p > 0.05 else 'NOT normal'}")
    normality_rows.append({"Group": f"Eco Class: {grp}", "W_statistic": round(stat, 4), "p_value": round(p, 6), "Normal": p > 0.05})

normality_table = pd.DataFrame(normality_rows)
print("\n-> Since p < 0.05 in (almost) every case, Total_loss is confirmed NOT normally")
print("   distributed, so non-parametric tests (Mann-Whitney U, Kruskal-Wallis H) are used.\n")

# Mann-Whitney U test: Total loss by District
print("=" * 65)
print("(i) MANN-WHITNEY U TEST -- Total Loss by District")
print("=" * 65)

bara = df.loc[df["District_lbl"] == "Bara", "Total_loss"]
parsa = df.loc[df["District_lbl"] == "Parsa", "Total_loss"]

district_summary = df.groupby("District_lbl")["Total_loss"].agg(["count", "median", "mean"]).round(2)
district_summary.columns = ["N", "Median", "Mean"]
print(district_summary, "\n")

u_stat, u_p = mannwhitneyu(bara, parsa, alternative="two-sided")
print(f"Mann-Whitney U = {u_stat:.3f}, p-value = {u_p:.4f}")
if u_p < 0.05:
    print("-> Reject H0 at 5% level: total loss significantly differs by district.\n")
else:
    print("-> Fail to reject H0 at 5% level: no significant difference by district.\n")

mw_result = pd.DataFrame([{
    "U_statistic": round(u_stat, 3), "p_value": round(u_p, 4), "Significant_at_5pct": u_p < 0.05
}])

# Kruskal-Wallis H test: Total loss by Economic Class
print("=" * 65)
print("(ii) KRUSKAL-WALLIS H TEST -- Total Loss by Economic Class")
print("=" * 65)

poor = df.loc[df["Eco_class_lbl"] == "Poor", "Total_loss"]
middle = df.loc[df["Eco_class_lbl"] == "Middle", "Total_loss"]
rich = df.loc[df["Eco_class_lbl"] == "Rich", "Total_loss"]

eco_summary = df.groupby("Eco_class_lbl")["Total_loss"].agg(["count", "median", "mean"]).round(2)
eco_summary.columns = ["N", "Median", "Mean"]
print(eco_summary, "\n")

h_stat, h_p = kruskal(poor, middle, rich)
print(f"Kruskal-Wallis H = {h_stat:.3f}, df = 2, p-value = {h_p:.4f}")
if h_p < 0.05:
    print("-> Reject H0 at 5% level: total loss significantly differs across economic classes.\n")
else:
    print("-> Fail to reject H0 at 5% level: no significant difference across economic classes.\n")

kw_result = pd.DataFrame([{
    "H_statistic": round(h_stat, 3), "df": 2, "p_value": round(h_p, 4), "Significant_at_5pct": h_p < 0.05
}])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    normality_table.to_excel(writer, sheet_name="Normality_Check", index=False)
    district_summary.to_excel(writer, sheet_name="District_Summary")
    mw_result.to_excel(writer, sheet_name="MannWhitney_District", index=False)
    eco_summary.to_excel(writer, sheet_name="EcoClass_Summary")
    kw_result.to_excel(writer, sheet_name="KruskalWallis_EcoClass", index=False)

print(f"All Q8 results saved to: {OUTPUT_FILE}")
