"""
Q3. Total valuation of loss of horti-agriculture products and livestock,
    and test of whether average loss per HH per year varies with:
      Economic Class x Main Occupation  -> Two-way ANOVA + post hoc

"""

import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

DATA_PATH = r"C:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"C:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q3_results.xlsx")


df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)


df["Total_loss"] = (
    df["Rice_price"] + df["Maize_price"] + df["Veg_price"] + df["Fruit_price"]
    + df["Cow.Price"] + df["Buf.Price"] + df["Gt.Price"]
)


df["Eco_class_lbl"] = df["Eco_class"].map({1: "Poor", 2: "Middle", 3: "Rich"})
df["Occupation_lbl"] = df["Main_Occupation"].map({1: "Agriculture", 2: "Service", 3: "Business"})

print("Overall Total Loss Valuation (Nrs/HH/year)")
overall_desc = df["Total_loss"].describe().round(2)
print(overall_desc, "\n")

group_table = df.groupby(["Eco_class_lbl", "Occupation_lbl"])["Total_loss"].agg(["count", "mean", "std"]).round(2)
group_table.columns = ["N", "Mean", "Std.Dev"]
print("Group means (Economic Class x Occupation):")
print(group_table, "\n")

# Two-way ANOVA
model = ols("Total_loss ~ C(Eco_class_lbl) * C(Occupation_lbl)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2).round(4)
print("Two-way ANOVA table:")
print(anova_table, "\n")

# Post hoc: Tukey HSD for each significant main effect ----
tukey_eco = pairwise_tukeyhsd(df["Total_loss"], df["Eco_class_lbl"], alpha=0.05)
tukey_eco_df = pd.DataFrame(data=tukey_eco._results_table.data[1:], columns=tukey_eco._results_table.data[0])
print("Post hoc (Tukey HSD) - Economic Class:")
print(tukey_eco, "\n")

tukey_occ = pairwise_tukeyhsd(df["Total_loss"], df["Occupation_lbl"], alpha=0.05)
tukey_occ_df = pd.DataFrame(data=tukey_occ._results_table.data[1:], columns=tukey_occ._results_table.data[0])
print("Post hoc (Tukey HSD) - Main Occupation:")
print(tukey_occ, "\n")

# Post hoc on the Eco_class x Occupation combination (useful if interaction is significant)
df["Combo"] = df["Eco_class_lbl"] + " / " + df["Occupation_lbl"]
tukey_combo = pairwise_tukeyhsd(df["Total_loss"], df["Combo"], alpha=0.05)
tukey_combo_df = pd.DataFrame(data=tukey_combo._results_table.data[1:], columns=tukey_combo._results_table.data[0])


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    overall_desc.to_frame(name="Total_loss").to_excel(writer, sheet_name="Overall_Descriptives")
    group_table.to_excel(writer, sheet_name="Group_Means")
    anova_table.to_excel(writer, sheet_name="TwoWay_ANOVA")
    tukey_eco_df.to_excel(writer, sheet_name="PostHoc_EcoClass", index=False)
    tukey_occ_df.to_excel(writer, sheet_name="PostHoc_Occupation", index=False)
    tukey_combo_df.to_excel(writer, sheet_name="PostHoc_Interaction", index=False)

print(f"All Q3 results saved in a single workbook: {OUTPUT_FILE}")
