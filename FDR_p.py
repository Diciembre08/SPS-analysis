import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


df = pd.read_csv('F:/psychology/psychology/SPS_oh_supplementary_analysis.csv')
target = 'sppn'
features = ['gender', 'Single', 'In a relationship', 'Engineering', 'Science', 'Arts and humanities',
            'Medicine', 'Stomach', 'Backache', 'Joint pain', 'Dysmenorrhea', 'Headache', 'Chest pain',
            'YEAR', 'AGE', 'HER', 'LBC', 'Interpersonal-difficulties', 'Academic pressure', 'Being punished',
            'Personal loss', 'Health and adaptability', 'Emotional abuse', 'Emotional neglect', 'Sexual abuse',
            'Physical abuse', 'Physical neglect']

results = []

for var in features:
    contingency = pd.crosstab(df[var], df[target])
    try:
        chi2, p, dof, expected = chi2_contingency(contingency)
    except Exception as e:
        print(f"variable {var} error: {e}")
        chi2, p = None, None
    results.append({'Variable': var, 'Chi2': chi2, 'p_value': p})


results_df = pd.DataFrame(results)
results_df = results_df.dropna(subset=['p_value'])
pvals = results_df['p_value'].values
bonferroni = multipletests(pvals, method='bonferroni')[1]
fdr = multipletests(pvals, method='fdr_bh')[1]

results_df['Bonferroni'] = bonferroni
results_df['FDR'] = fdr
results_df = results_df.reset_index(drop=True)

print(results_df)
results_df.to_csv("F:/psychology/psychology/chi2_results_corrected.csv", index=False)