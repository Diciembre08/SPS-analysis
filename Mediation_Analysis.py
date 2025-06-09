import pandas as pd
from pingouin import mediation_analysis


path = ''
data = pd.read_csv(path)
X = 'Sexual abuse'
Y = 'sppn'
mediators = [col for col in data.columns if col not in [X, Y]]
all_results = []


for mediator in mediators:
    try:
        result = mediation_analysis(data=data, x=X, m=mediator, y=Y, alpha=0.05, seed=42)
        result.insert(0, 'mediator', mediator)
        all_results.append(result)
    except Exception as e:
        print(f"Failed for mediator: {mediator} - {e}")


mediation_df = pd.concat(all_results, ignore_index=True)
mediation_df.head()
print(mediation_df)
mediation_df.to_csv('Sexual_abuse_mediation_results.csv', index=False)