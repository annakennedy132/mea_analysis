import pandas as pd
import statsmodels.formula.api as smf

# Load the dataset
data = pd.read_csv(r'c:\Users\Windows\Documents\Multi Channel Systems\Multi Channel Analyzer\global_data.csv')

# Check the first few rows
print(data.head())

print(data['LightLevel'].value_counts())
print(data['Frequency'].value_counts())
print(data['Group'].value_counts())

model = smf.mixedlm("Amplitude ~ Group + LightLevel + Frequency", 
                     data, 
                     groups=data["Experiment"], 
                     re_formula="~Channel")

# Fit the model and save the results
result = model.fit()

# Show the results
print(result.summary())

import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot to visualize amplitude differences by group
plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Amplitude', data=data)
plt.title('Amplitude by Group')
plt.ylabel('Amplitude')
plt.xlabel('Group')
plt.ylim(0, data['Amplitude'].max() * 1.1)  # Adjust the y-axis limit for better visibility
plt.show()