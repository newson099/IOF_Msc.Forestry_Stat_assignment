"""
Q13. Binary Logistic Regression: Determining factors for involvement in
     most of the Buffer Zone (BZ) activities.

     Dependent variable   (Y) = Invo_men  (0 = Not involved, 1 = Involved)
     Independent variables(X) = socio-economic variables:
         Continuous : Age, Education, Family size, LSU,
                       Land holding (ropani), Distance from BZ
         Categorical: District, Sex of HH Head, Ethnicity,
                       Economic Class, Main Occupation

"""

import os
import numpy as np
import pandas as pd
from statsmodels.formula.api import logit

DATA_PATH = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\spss.csv"
OUTPUT_DIR = r"E:\IOF\!st yr 2nd sem\stat\001_Internal_assessment\ass\13"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q13_results.xlsx")

ALPHA = 0.05

# ---- 1. Load and clean ----
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

# ---- 2. Attach categorical labels ----
df["District_lbl"] = df["District"].map({1: "Bara", 2: "Parsa"})
df["Sex_lbl"] = df["HH_Head_sex"].map({1: "Female", 2: "Male"})
df["Ethnic_lbl"] = df["Ehnicity"].map({1: "Dalit", 2: "Indigenous", 3: "Madhesi", 4: "Brahmin_Chhetri"})
df["Eco_class_lbl"] = df["Eco_class"].map({1: "Poor", 2: "Middle", 3: "Rich"})
df["Occupation_lbl"] = df["Main_Occupation"].map({1: "Agriculture", 2: "Service", 3: "Business"})

print("Dependent variable (Invo_men) distribution:")
print(df["Invo_men"].value_counts(), "\n")

formula = (
    "Invo_men ~ Age + Q('Edu.Hh') + Family_size + LSU + Land_holding_ropani + Dis_from "
    "+ C(District_lbl) + C(Sex_lbl) + C(Ethnic_lbl) + C(Eco_class_lbl) + C(Occupation_lbl)"
)
model = logit(formula, data=df).fit()
print(model.summary())

coef_table = pd.DataFrame({
    "Coefficient (log-odds)": model.params.round(4),
    "Std_Error": model.bse.round(4),
    "z_stat": model.tvalues.round(3),
    "p_value": model.pvalues.round(4),
    "Odds_Ratio": np.exp(model.params).round(3),
    "OR_CI_lower_95%": np.exp(model.conf_int()[0]).round(3),
    "OR_CI_upper_95%": np.exp(model.conf_int()[1]).round(3),
    "Significant_5pct": model.pvalues < ALPHA,
})
print("\n", coef_table.to_string())

pred_prob = model.predict(df)
pred_class = (pred_prob >= 0.5).astype(int)
accuracy = (pred_class == df["Invo_men"]).mean()

model_summary = pd.DataFrame([{
    "N": int(model.nobs),
    "Pseudo_R2_McFadden": round(model.prsquared, 4),
    "Log_Likelihood": round(model.llf, 3),
    "LL_Null": round(model.llnull, 3),
    "LR_chi2": round(model.llr, 3),
    "LR_p_value": round(model.llr_pvalue, 4),
    "Classification_Accuracy": round(accuracy, 4),
}])
print("\nModel Summary:")
print(model_summary.to_string(index=False))

if model.llr_pvalue < ALPHA:
    print(f"\n-> Reject H0 (overall model): the model is significant at {int(ALPHA*100)}% level.")
else:
    print(f"\n-> Fail to reject H0 (overall model): not significant at {int(ALPHA*100)}% level.")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    coef_table.to_excel(writer, sheet_name="Logistic_Coefficients")
    model_summary.to_excel(writer, sheet_name="Model_Summary", index=False)

print(f"\nAll Q13 results saved to: {OUTPUT_FILE}")
