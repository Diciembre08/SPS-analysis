import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt


file_path = r"F:\psychology\psychology\missing_value.xlsx"
df = pd.read_excel(file_path)
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title("Missing Data Matrix - No Missing Values")
plt.show()