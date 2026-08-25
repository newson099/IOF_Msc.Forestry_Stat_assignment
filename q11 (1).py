"""
Q11. Multiple OLS Regression:
        Dependent variable   (Y) = Total Loss (Nrs/HH/year)  -- cube-root
                                    transformed to reduce extreme skewness
        Independent variables (X) = socio-economic variables:
            Continuous : Age, Education, Family size, LSU,
                          Land holding (ropani), Distance from BZ
            Categorical: District, Sex of HH Head, Ethnicity,
                          Economic Class, Main Occupation

     Diagnostics (all at 10% level of significance, as specified):
        - Multicollinearity  -> Variance Inflation Factor (VIF)
        - Heteroscedasticity -> Modified (studentized) Breusch-Pagan test
        - Normality of residuals -> Jarque-Bera and Shapiro-Wilk tests

"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro

DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q11_results.xlsx")

ALPHA = 0.10  # 10% level of significance, as specified in the question

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

df["Total_loss"] = (
    df["Rice_price"] + df["Maize_price"] + df["Veg_price"] + df["Fruit_price"]
    + df["Cow.Price"] + df["Buf.Price"] + df["Gt.Price"]
)

df["Total_loss_cbrt"] = np.cbrt(df["Total_loss"])
print(f"Skewness before transformation: {df['Total_loss'].skew():.3f}")
print(f"Skewness after cube-root transformation: {df['Total_loss_cbrt'].skew():.3f}\n")

df["District_lbl"] = df["District"].map({1: "Bara", 2: "Parsa"})
df["Sex_lbl"] = df["HH_Head_sex"].map({1: "Female", 2: "Male"})
df["Ethnic_lbl"] = df["Ehnicity"].map({1: "Dalit", 2: "Indigenous", 3: "Madhesi", 4: "Brahmin_Chhetri"})
df["Eco_class_lbl"] = df["Eco_class"].map({1: "Poor", 2: "Middle", 3: "Rich"})
df["Occupation_lbl"] = df["Main_Occupation"].map({1: "Agriculture", 2: "Service", 3: "Business"})

# Fit the multiple OLS regression 
formula = (
    "Total_loss_cbrt ~ Age + Q('Edu.Hh') + Family_size + LSU + Land_holding_ropani + Dis_from "
    "+ C(District_lbl) + C(Sex_lbl) + C(Ethnic_lbl) + C(Eco_class_lbl) + C(Occupation_lbl)"
)
model = ols(formula, data=df).fit()
print(model.summary())

coef_table = pd.DataFrame({
    "Coefficient": model.params.round(4),
    "Std_Error": model.bse.round(4),
    "t_stat": model.tvalues.round(3),
    "p_value": model.pvalues.round(4),
    "Significant_10pct": model.pvalues < ALPHA,
})

#  Multicollinearity (VIF)
print("\n" + "=" * 65)
print("MULTICOLLINEARITY CHECK -- Variance Inflation Factor (VIF)")
print("=" * 65)

X_design = model.model.exog
X_names = model.model.exog_names
vif_data = pd.DataFrame({
    "Variable": X_names,
    "VIF": [variance_inflation_factor(X_design, i) for i in range(X_design.shape[1])],
}).round(3)
print(vif_data.to_string(index=False))
print("\nRule of thumb: VIF > 10 indicates serious multicollinearity.\n")

# Heteroscedasticity Modified (studentized) Breusch-Pagan test
print("=" * 65)
print("HETEROSCEDASTICITY CHECK -- Modified (Studentized) Breusch-Pagan Test")
print("=" * 65)
print("H0: Residuals have constant variance (homoscedasticity)")

bp_lm_stat, bp_lm_p, bp_f_stat, bp_f_p = het_breuschpagan(model.resid, model.model.exog)
print(f"LM statistic = {bp_lm_stat:.3f}, LM p-value = {bp_lm_p:.4f}")
print(f"F statistic  = {bp_f_stat:.3f}, F p-value  = {bp_f_p:.4f}")
if bp_lm_p < ALPHA:
    print(f"-> Reject H0 at {int(ALPHA*100)}% level: heteroscedasticity IS present.\n")
else:
    print(f"-> Fail to reject H0 at {int(ALPHA*100)}% level: no evidence of heteroscedasticity.\n")

bp_result = pd.DataFrame([{
    "LM_statistic": round(bp_lm_stat, 3), "LM_p_value": round(bp_lm_p, 4),
    "F_statistic": round(bp_f_stat, 3), "F_p_value": round(bp_f_p, 4),
    "Alpha": ALPHA, "Heteroscedastic": bp_lm_p < ALPHA,
}])

# Normality of residuals
print("=" * 65)
print("NORMALITY CHECK -- Jarque-Bera and Shapiro-Wilk Tests on Residuals")
print("=" * 65)
print("H0: Residuals are normally distributed")

jb_stat, jb_p, skew, kurt = jarque_bera(model.resid)
print(f"Jarque-Bera: JB = {jb_stat:.3f}, p-value = {jb_p:.4f}  (skew={skew:.3f}, kurtosis={kurt:.3f})")

sw_stat, sw_p = shapiro(model.resid)
print(f"Shapiro-Wilk: W = {sw_stat:.4f}, p-value = {sw_p:.4f}")

if jb_p < ALPHA:
    print(f"-> Reject H0 at {int(ALPHA*100)}% level: residuals are NOT normally distributed.\n")
else:
    print(f"-> Fail to reject H0 at {int(ALPHA*100)}% level: residuals appear normally distributed.\n")

normality_result = pd.DataFrame([{
    "Jarque_Bera_stat": round(jb_stat, 3), "JB_p_value": round(jb_p, 4),
    "Shapiro_W": round(sw_stat, 4), "Shapiro_p_value": round(sw_p, 4),
    "Alpha": ALPHA, "Normal_residuals": jb_p >= ALPHA,
}])

model_summary = pd.DataFrame([{
    "N": int(model.nobs),
    "R_squared": round(model.rsquared, 4),
    "Adj_R_squared": round(model.rsquared_adj, 4),
    "F_statistic": round(model.fvalue, 3),
    "F_p_value": round(model.f_pvalue, 4),
}])

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    coef_table.to_excel(writer, sheet_name="Coefficients")
    model_summary.to_excel(writer, sheet_name="Model_Summary", index=False)
    vif_data.to_excel(writer, sheet_name="VIF_Multicollinearity", index=False)
    bp_result.to_excel(writer, sheet_name="BreuschPagan_Heterosced", index=False)
    normality_result.to_excel(writer, sheet_name="Normality_Residuals", index=False)

print(f"All Q11 results saved to: {OUTPUT_FILE}")
