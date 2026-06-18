# --- Approach ---
# The process combines rule-based classification using qualitative marker
# readings (positives (+) or negatives (-)) from available cell type
# definitions to assign cells to defined identities based on matching to
# known identity patterns. This step can give a good first pass at assigning
# defined identities to most of the "known-match" cells. The identity of
# unassigned cells will be "Unknown" or "Ambiguous".
#
# Because many of the marker proteins are membrane-bound surface proteins,
# I compared the mean expression levels of the marker proteins between the
# cell and membrane compartments to potentially improve the coverage of
# assignments. These average values provide measures of the cell's
# quantitative expression level of a marker. Then, I used the confidently
# labeled cells from the rule-based identity assignment as the training
# labels for a Random Forest classifier to learn the pattern of cell
# populations in the quantitative expression data.
#
# Finally, I set a threshold for the trained classifier model and only used
# classifier predictions with a measurable level of confidence. This hybrid
# approach uses qualitative identification for precision and quantitative
# modeling for coverage to allow identification of more cells without
# relying only on possibly inaccurate manual cell assignments.

# --- Key Findings ---
# The rule-based stage was able to classify a substantial subset of cells
# with high confidence, but some cells remained "Unknown" or "Ambiguous"
# because their qualitative marker patterns were incomplete, noisy, or
# overlapping across multiple cell types. This showed that qualitative
# marker information alone was useful, but not sufficient for maximizing
# annotation coverage.
#
# Adding quantitative expression data improved the annotation process by
# giving the model more information than the +/- labels alone. The Random
# Forest classifier showed strong performance on the confidently labeled
# cells and was able to recover additional cells that were not assigned in
# the initial rule-based stage. At the same time, agreement between the
# model predictions and the original confident rule-based labels remained
# high, which suggests that the quantitative refinement was biologically
# consistent with the known marker definitions.
#
# By applying a confidence threshold, the final method favored precision
# while still increasing the total number of annotated cells. Overall,
# combining qualitative rules with quantitative modeling gave broader and
# more robust cell annotation than either method alone.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# load data
df = pd.read_csv("../Q3_gene_expression_data.csv", index_col=0)
print(f"Loaded {len(df)} cells with {len(df.columns)} columns")

# cell type marker definitions from Q3_cell_type_markers.txt
cell_type_markers = {
    "Arteries":           {"CD31": "+", "TCR": "-", "CD117": "-", "CD11b": "-"},
    "B Cells":            {"CD19": "+", "B220": "+"},
    "Capillaries":        {"Endomucin": "+", "TCR": "-", "CD117": "-", "CD11b": "-"},
    "CD8 T Cells":        {"CD3": "+", "CD8": "+", "TCR": "+", "CD4": "-"},
    "CFU-E":              {"CD117": "+", "CD71": "+", "Ter119": "-", "CD41": "-"},
    "Endothelial Niche":  {"CD105": "+", "CD45": "-", "Ter119": "-"},
    "Erythroblasts":      {"Ter119": "+", "CD45": "-", "CD71": "+"},
    "Erythrocytes":       {"Ter119": "+", "CD45": "-", "CD71": "-"},
    "Neutrophils":        {"CD11b": "+", "Ly6g": "+", "F4_80": "-"},
    "Perivascular Niche": {"Lepr": "+", "CD45": "-", "Ter119": "-", "CD31": "-"},
}

qual_columns = ["Ter119", "CD45", "CD71", "CD11b", "TCR", "CD117", "CD31",
                "CD19", "CD105", "Endomucin", "Lepr", "B220", "CD3", "CD4",
                "CD8", "F4_80", "Ly6g", "CD41"]


# ---- Stage 1: classify using qualitative +/- labels ----

def get_sign(value):
    # pull the + or - from labels like "CD45+" or "Ter119-"
    if str(value).endswith("+"):
        return "+"
    elif str(value).endswith("-"):
        return "-"
    return None


def classify_by_rules(row):
    # check each cell type definition against this cell's labels
    matches = []
    for cell_type, markers in cell_type_markers.items():
        all_match = True
        for marker, expected in markers.items():
            if get_sign(row[marker]) != expected:
                all_match = False
                break
        if all_match:
            matches.append(cell_type)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return "Ambiguous"  # matched more than one cell type
    return "Unknown"


print("\n-- Stage 1: Qualitative rule-based classification --")
df["stage1"] = df.apply(classify_by_rules, axis=1)

stage1_counts = df["stage1"].value_counts()
print(stage1_counts.to_string())
stage1_classified = df["stage1"].isin(cell_type_markers.keys()).sum()
print(f"Classified: {stage1_classified} / {len(df)} ({stage1_classified / len(df) * 100:.1f}%)")


# ---- Stage 2: use quantitative data to improve with Random Forest ----

# grab the markers used in our cell type definitions
relevant_markers = set()
for markers in cell_type_markers.values():
    relevant_markers.update(markers.keys())

# pick quantitative features - Cell Mean and Membrane Mean for each marker
# these are the most useful since most markers are surface proteins
quant_features = []
for marker in sorted(relevant_markers):
    for compartment in ["Cell", "Membrane"]:
        col = f"{marker}: {compartment}: Mean"
        if col in df.columns:
            quant_features.append(col)

print(f"\nUsing {len(quant_features)} quantitative features")

# train on the cells that stage 1 confidently classified
confident_mask = df["stage1"].isin(cell_type_markers.keys())
X_train = df.loc[confident_mask, quant_features].values
y_train = df.loc[confident_mask, "stage1"].values
X_all = df[quant_features].values

# fill any missing values with 0
X_train = np.nan_to_num(X_train)
X_all = np.nan_to_num(X_all)

print(f"Training on {len(X_train)} confidently labeled cells")

# train random forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# check how good the model is with cross validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# predict on all cells
probas = rf.predict_proba(X_all)
predictions = rf.predict(X_all)
max_confidence = np.max(probas, axis=1)


# ---- Stage 3: only keep predictions where model is confident ----

final_labels = []
for i in range(len(df)):
    if max_confidence[i] >= 0.6:
        final_labels.append(predictions[i])
    else:
        final_labels.append("Unknown")

df["final_label"] = final_labels

print("\n-- Final Classification (after quantitative refinement) --")
final_counts = df["final_label"].value_counts()
print(final_counts.to_string())
final_classified = (df["final_label"] != "Unknown").sum()
print(f"Classified: {final_classified} / {len(df)} ({final_classified / len(df) * 100:.1f}%)")

# check agreement between stage 1 and final
both_classified = df[confident_mask]
agreement = (both_classified["stage1"] == both_classified["final_label"]).sum()
print(f"Agreement with Stage 1: {agreement}/{len(both_classified)} ({agreement / len(both_classified) * 100:.1f}%)")

# how many unknown cells did the model figure out?
was_unknown = df["stage1"].isin(["Unknown", "Ambiguous"])
now_classified = (df.loc[was_unknown, "final_label"] != "Unknown").sum()
print(f"Cells rescued from Unknown/Ambiguous: {now_classified}/{was_unknown.sum()}")

# top features
print("\nTop 10 most important features:")
importances = rf.feature_importances_
feat_imp = sorted(zip(quant_features, importances), key=lambda x: x[1], reverse=True)
for feat, imp in feat_imp[:10]:
    print(f"  {feat}: {imp:.4f}")


# ---- Plots ----

# stage 1 results
plt.figure(figsize=(8, 5))
s1 = stage1_counts.sort_values()
plt.barh(s1.index, s1.values)
plt.title("Stage 1: Qualitative Rule-Based Classification")
plt.xlabel("Number of Cells")
plt.savefig("Q3_stage1.png")
plt.show()

# final results
plt.figure(figsize=(8, 5))
fc = final_counts.sort_values()
plt.barh(fc.index, fc.values, color="orange")
plt.title("Final Classification (After Quantitative Refinement)")
plt.xlabel("Number of Cells")
plt.savefig("Q3_final.png")
plt.show()

# confidence distribution
plt.figure(figsize=(8, 5))
plt.hist(max_confidence, bins=50)
plt.axvline(0.6, color="red", linestyle="--", label="Confidence threshold (0.6)")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Number of Cells")
plt.legend()
plt.savefig("Q3_confidence.png")
plt.show()

print("\nPlots saved to Q3_stage1.png, Q3_final.png, Q3_confidence.png")
