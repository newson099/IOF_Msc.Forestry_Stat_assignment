
import os
import pandas as pd


DATA_PATH = r"C:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Data.csv"
OUTPUT_DIR = r"C:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment"


df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

df["District"] = df["District"].replace("2+C182:BC182", "2").astype(int)


df["District"] = df["District"].map({1: "Bara", 2: "Parsa"})
df["HH_Head_sex"] = df["HH_Head_sex"].map({1: "Female", 2: "Male"})
df["Ehnicity"] = df["Ehnicity"].map({1: "Dalit", 2: "Indigenous", 3: "Madhesi", 4: "Brahmin/Chhetri"})
df["Eco_class"] = df["Eco_class"].map({1: "Poor", 2: "Middle", 3: "Rich"})
df["Main_Occupation"] = df["Main_Occupation"].map({1: "Agriculture", 2: "Service", 3: "Business"})

# Categorical variables: frequency and percentage
categorical_vars = ["District", "HH_Head_sex", "Ehnicity", "Eco_class", "Main_Occupation"]

print("CATEGORICAL VARIABLES\n")
for var in categorical_vars:
    counts = df[var].value_counts()
    percent = df[var].value_counts(normalize=True) * 100
    table = pd.DataFrame({"Frequency": counts, "Percent": percent.round(1)})
    print(f"-- {var} --")
    print(table, "\n")

    out_path = os.path.join(OUTPUT_DIR, f"q1_freq_{var}.csv")
    table.to_csv(out_path)
    print(f"Saved -> {out_path}\n")

# Continuous variables: mean, std, min, max, etc. ----
continuous_vars = ["Age", "Edu.Hh", "Family_size", "LSU", "Land_holding_ropani", "Dis_from"]

summary = df[continuous_vars].describe().round(2).T
print("CONTINUOUS VARIABLES\n")
print(summary)

out_path = os.path.join(OUTPUT_DIR, "q1_continuous_summary.csv")
summary.to_csv(out_path)
print(f"\nSaved -> {out_path}")
