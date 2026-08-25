"""
Q9. Karl Pearson's and Spearman's rank correlation matrix (with significance
    tests) between:
      1. Total Forest Product Income (Nrs)
      2. Total Horti-agriculture Loss valuation (Nrs)
      3. Total Livestock (LS) Loss valuation (Nrs)
      4. Distance of Buffer Zone from HH (min)
"""

import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q9_results.xlsx")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

df["Forest_income"] = (
    df["Fuel_qty"] * df["Fuel_price"]
    + df["Grass_qty"] * df["Grass_price"]
    + df["Leaf_qty"] * df["Leaf_pric"]
)

df["Agri_loss"] = df["Rice_price"] + df["Maize_price"] + df["Veg_price"] + df["Fruit_price"]

df["LS_loss"] = df["Cow.Price"] + df["Buf.Price"] + df["Gt.Price"]

# Distance of BZ from household
df["Distance"] = df["Dis_from"]

variables = ["Forest_income", "Agri_loss", "LS_loss", "Distance"]
labels = {
    "Forest_income": "Forest Product Income",
    "Agri_loss": "Horti-agriculture Loss",
    "LS_loss": "Livestock Loss",
    "Distance": "Distance from BZ",
}

# correlation matrix + significance (p-value) matrix 


def correlation_with_significance(data, cols, method):
    """Return (r-matrix, p-matrix) for either 'pearson' or 'spearman'."""
    corr_func = pearsonr if method == "pearson" else spearmanr
    r_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for row in cols:
        for col in cols:
            r, p = corr_func(data[row], data[col])
            r_matrix.loc[row, col] = round(r, 3)
            p_matrix.loc[row, col] = round(p, 4)

    r_matrix.index = r_matrix.columns = [labels[c] for c in cols]
    p_matrix.index = p_matrix.columns = [labels[c] for c in cols]
    return r_matrix, p_matrix


pearson_r, pearson_p = correlation_with_significance(df, variables, "pearson")
spearman_r, spearman_p = correlation_with_significance(df, variables, "spearman")

print("KARL PEARSON'S CORRELATION MATRIX (r)")
print(pearson_r, "\n")
print("Pearson significance (p-values)")
print(pearson_p, "\n")

print("SPEARMAN'S RANK CORRELATION MATRIX (rho)")
print(spearman_r, "\n")
print("Spearman significance (p-values)")
print(spearman_p, "\n")

#  summary table: each pair, both r and rho, with significance decision
pairs = []
for i, v1 in enumerate(variables):
    for v2 in variables[i + 1:]:
        r_p, p_p = pearsonr(df[v1], df[v2])
        r_s, p_s = spearmanr(df[v1], df[v2])
        pairs.append({
            "Variable 1": labels[v1],
            "Variable 2": labels[v2],
            "Pearson_r": round(r_p, 3),
            "Pearson_p": round(p_p, 4),
            "Pearson_Significant_5pct": p_p < 0.05,
            "Spearman_rho": round(r_s, 3),
            "Spearman_p": round(p_s, 4),
            "Spearman_Significant_5pct": p_s < 0.05,
        })

pairwise_summary = pd.DataFrame(pairs)
print("Pairwise summary (both methods):")
print(pairwise_summary.to_string(index=False))

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    pearson_r.to_excel(writer, sheet_name="Pearson_r")
    pearson_p.to_excel(writer, sheet_name="Pearson_p_values")
    spearman_r.to_excel(writer, sheet_name="Spearman_rho")
    spearman_p.to_excel(writer, sheet_name="Spearman_p_values")
    pairwise_summary.to_excel(writer, sheet_name="Pairwise_Summary", index=False)

print(f"\nAll Q9 results saved to: {OUTPUT_FILE}")
