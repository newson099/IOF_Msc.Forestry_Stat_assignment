"""
Q5. Objective: To test whether there is an association between distance-based user category
(Near <20 min, Middle 20–40 min, Far >40 min) and responses on risk category for two illegal activities in the park,
Encroachment and Wildlife Poaching, using the Chi-square test of independence of attributes at 10% level of significance.

"""

import os
import pandas as pd
from scipy.stats import chi2_contingency

DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q5_results.xlsx")

ALPHA = 0.10  # 10% level of significance, as specified in the question

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)


def categorize_distance(minutes):
    if minutes < 20:
        return "Near User (<20 min)"
    elif minutes <= 40:
        return "Middle User (20-40 min)"
    else:
        return "Far User (>40 min)"


df["User_category"] = df["Dis_from"].apply(categorize_distance)
cat_order = ["Near User (<20 min)", "Middle User (20-40 min)", "Far User (>40 min)"]

print("Distance-based User Category counts:")
print(df["User_category"].value_counts().reindex(cat_order), "\n")

risk_labels = {1: "Very Less", 2: "Less", 3: "High"}
df["Encroachment_risk"] = df["Ill_encroachment"].map(risk_labels)
df["Poaching_risk"] = df["Ill_wild_poaching"].map(risk_labels)
risk_order = ["Very Less", "Less", "High"]


def run_chi_square(data, row_var, col_var, row_order, col_order, alpha=ALPHA):
    """Build contingency table and run chi-square test of independence."""
    table = pd.crosstab(data[row_var], data[col_var])
    table = table.reindex(index=row_order, columns=col_order)

    chi2, p_value, dof, expected = chi2_contingency(table)
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns).round(2)

    print(f"Contingency table: {row_var} x {col_var}")
    print(table, "\n")
    print("Expected frequencies:")
    print(expected_df, "\n")
    print(f"Chi-square = {chi2:.3f}, df = {dof}, p-value = {p_value:.4f}")
    if p_value < alpha:
        print(f"-> Reject H0 at {int(alpha*100)}% level: significant association exists.\n")
    else:
        print(f"-> Fail to reject H0 at {int(alpha*100)}% level: no significant association.\n")

    result = pd.DataFrame([{
        "Chi_square": round(chi2, 3),
        "df": dof,
        "p_value": round(p_value, 4),
        "Alpha": alpha,
        "Significant": p_value < alpha,
    }])
    return table, expected_df, result


#  Chi-square: User category vs Encroachment risk 
print("=" * 65)
print("(a) USER CATEGORY vs ENCROACHMENT RISK")
print("=" * 65)
enc_table, enc_expected, enc_result = run_chi_square(
    df, "User_category", "Encroachment_risk", cat_order, risk_order
)

#  Chi-square: User category vs Wildlife Poaching risk 
print("=" * 65)
print("(b) USER CATEGORY vs WILDLIFE POACHING RISK")
print("=" * 65)
poach_table, poach_expected, poach_result = run_chi_square(
    df, "User_category", "Poaching_risk", cat_order, risk_order
)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    enc_table.to_excel(writer, sheet_name="Encroachment_Observed")
    enc_expected.to_excel(writer, sheet_name="Encroachment_Expected")
    enc_result.to_excel(writer, sheet_name="Encroachment_ChiSquare", index=False)

    poach_table.to_excel(writer, sheet_name="Poaching_Observed")
    poach_expected.to_excel(writer, sheet_name="Poaching_Expected")
    poach_result.to_excel(writer, sheet_name="Poaching_ChiSquare", index=False)

print(f"All Q5 results saved to: {OUTPUT_FILE}")
