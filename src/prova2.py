from enigmatoolbox.datasets import load_summary_stats
import pandas as pd

# Load summary statistics for ENIGMA-Antisocial Behavior
sum_stats = load_summary_stats('depression')

# Get case-control cortical thickness table
CT = sum_stats['CortThick_case_vs_controls_adolescent']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(CT)
print(CT['fdr_p'].to_string(index=False))




