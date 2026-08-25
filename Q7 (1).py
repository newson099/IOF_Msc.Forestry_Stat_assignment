"""
Q7. Are people's rankings of HWC (Human-Wildlife Conflict) mitigation
    activities different on average? Tested with the Friedman ANOVA test
    
"""

import os
import pandas as pd
from scipy.stats import friedmanchisquare

DATA_PATH = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output\Data.csv"
OUTPUT_DIR = r"c:\Users\anews\OneDrive\Desktop\Stat_Newson_assign\Py_CodeForStat_assignment\Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "q7_results.xlsx")


df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# The 6 HWC mitigation activities being ranked (1 = most effective)
activities = {
    "Rank_Relief_dist": "Relief Distribution",
    "Rank_Training_awareness_rank": "Training / Awareness",
    "Rank_Bio_fencing_rank": "Bio-fencing",
    "Rank_Electrical_fenncing_rank": "Electric Fencing",
    "Rank_trenching": "Trenching",
    "Rank_Deterrent_rank": "Deterrents",
}

# Average rank of each activity (lower = perceived more effective)
avg_ranks = df[list(activities.keys())].mean().round(2)
avg_ranks.index = [activities[c] for c in avg_ranks.index]
avg_ranks = avg_ranks.sort_values()

print("Average rank given to each HWC mitigation activity (1 = most effective):")
print(avg_ranks, "\n")

# Friedman ANOVA test
# H0: all activities are ranked the same on average (no real preference)
# H1: at least one activity is ranked differently from the others
columns_in_order = list(activities.keys())
stat, p_value = friedmanchisquare(*[df[c] for c in columns_in_order])

print(f"Friedman test: chi-square = {stat:.3f}, df = {len(activities) - 1}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("-> Reject H0 at 5% level: people rank the activities differently.\n")
else:
    print("-> Fail to reject H0 at 5% level: no significant difference in rankings.\n")

result_table = pd.DataFrame([{
    "N_households": len(df),
    "N_activities": len(activities),
    "Chi_square": round(stat, 3),
    "df": len(activities) - 1,
    "p_value": round(p_value, 4),
    "Significant_at_5pct": p_value < 0.05,
}])


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    avg_ranks.to_frame(name="Average_Rank").to_excel(writer, sheet_name="Average_Ranks")
    result_table.to_excel(writer, sheet_name="Friedman_Test", index=False)

print(f"All Q7 results saved to: {OUTPUT_FILE}")
