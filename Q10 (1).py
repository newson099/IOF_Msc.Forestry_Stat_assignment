"""
Q10. Simple Linear Regression:
        Dependent variable   (Y) = Total Loss (Nrs/HH/year)
        Independent variable (X) = Distance of BZ from HH (min)
     Includes the regression line plot and a significance test of the
     regression coefficient (slope).

"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm


DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q10_results.xlsx")
PLOT_FILE = os.path.join(OUTPUT_DIR, "q10_regression_plot.png")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)

df["Total_loss"] = (
    df["Rice_price"] + df["Maize_price"] + df["Veg_price"] + df["Fruit_price"]
    + df["Cow.Price"] + df["Buf.Price"] + df["Gt.Price"]
)

X = df["Dis_from"]      # independent variable
Y = df["Total_loss"]    # dependent variable

# Fit the simple linear regression: Total_loss = a + b*Distance 
X_with_const = sm.add_constant(X)  # adds the intercept term
model = sm.OLS(Y, X_with_const).fit()

print(model.summary())

intercept, slope = model.params
t_stat_slope = model.tvalues["Dis_from"]
p_value_slope = model.pvalues["Dis_from"]
r_squared = model.rsquared

print(f"\nRegression equation: Total_loss = {intercept:.2f} + ({slope:.2f}) x Distance")
print(f"Slope (b) = {slope:.3f},  t = {t_stat_slope:.3f},  p-value = {p_value_slope:.4f}")
print(f"R-squared = {r_squared:.4f}  (i.e. Distance explains {r_squared*100:.1f}% of the variation in Total loss)")
if p_value_slope < 0.05:
    print("-> Reject H0 at 5% level: Distance is a significant predictor of Total loss.\n")
else:
    print("-> Fail to reject H0 at 5% level: Distance is NOT a significant predictor.\n")

#  Build a clean results table for the coefficient test 
coef_table = pd.DataFrame({
    "Coefficient": model.params.round(3),
    "Std_Error": model.bse.round(3),
    "t_stat": model.tvalues.round(3),
    "p_value": model.pvalues.round(4),
    "CI_lower_95%": model.conf_int()[0].round(3),
    "CI_upper_95%": model.conf_int()[1].round(3),
})
coef_table.index = ["Intercept (a)", "Distance (b)"]

model_summary = pd.DataFrame([{
    "N": int(model.nobs),
    "R_squared": round(r_squared, 4),
    "Adj_R_squared": round(model.rsquared_adj, 4),
    "F_statistic": round(model.fvalue, 3),
    "F_p_value": round(model.f_pvalue, 4),
}])

# Plot: scatter of actual data + fitted regression line 
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.5, color="steelblue", label="Observed households")
x_line = pd.Series(sorted(X.unique()))
y_line = intercept + slope * x_line
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Fitted line: Y = {intercept:.1f} + {slope:.1f}X")
plt.xlabel("Distance of BZ from HH (minutes)")
plt.ylabel("Total Loss (Nrs/HH/year)")
plt.title("Simple Linear Regression: Total Loss vs Distance from BZ")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
print(f"Regression plot saved to: {PLOT_FILE}")

#  results 
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    coef_table.to_excel(writer, sheet_name="Coefficients")
    model_summary.to_excel(writer, sheet_name="Model_Summary", index=False)

print(f"All Q10 results saved to: {OUTPUT_FILE}")
