import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('telco_churn.csv')

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.head())

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert Churn to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"\nRows after cleaning: {len(df)}")
print(f"Churn Rate: {df['Churn'].mean()*100:.2f}%")

# Plot 1 - Churn by Contract Type
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('Churn by Contract Type', fontsize=14)
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(labels=['Retained', 'Churned'])
plt.tight_layout()
plt.savefig('churn_by_contract.png', dpi=150)
plt.close()
print("Plot 1 saved")

# Plot 2 - Tenure Distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette='Set2')
plt.title('Tenure vs Churn', fontsize=14)
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('tenure_churn.png', dpi=150)
plt.close()
print("Plot 2 saved")

# Plot 3 - Monthly Charges Boxplot
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette='Set2')
plt.title('Monthly Charges: Churned vs Retained', fontsize=14)
plt.xticks([0,1], ['Retained', 'Churned'])
plt.tight_layout()
plt.savefig('charges_churn.png', dpi=150)
plt.close()
print("Plot 3 saved")

# Plot 4 - Correlation Heatmap
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("Plot 4 saved")

# Plot 5 - Churn by Internet Service
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set2')
plt.title('Churn by Internet Service Type', fontsize=14)
plt.xlabel('Internet Service')
plt.ylabel('Number of Customers')
plt.legend(labels=['Retained', 'Churned'])
plt.tight_layout()
plt.savefig('churn_by_internet.png', dpi=150)
plt.close()
print("Plot 5 saved")

# Plot 6 - Churn by Tenure Group
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12 months','13-24 months','25-48 months','49-72 months'])
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='tenure_group', hue='Churn', palette='Set2')
plt.title('Churn by Tenure Group', fontsize=14)
plt.xlabel('Tenure Group')
plt.ylabel('Number of Customers')
plt.legend(labels=['Retained', 'Churned'])
plt.tight_layout()
plt.savefig('churn_by_tenure_group.png', dpi=150)
plt.close()
print("Plot 6 saved")