import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

print("--- Dataset 1: Unemployment in India ---")
print(df1.head())
print(df1.info())

print("\n--- Dataset 2: Unemployment Rate upto 11/2020 ---")
print(df2.head())
print(df2.info())

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Drop rows where Region is null in df1
df1 = df1.dropna(subset=['Region'])

# Convert Date to datetime
df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)

print("df1 Date Range:", df1['Date'].min(), "to", df1['Date'].max())
print("df2 Date Range:", df2['Date'].min(), "to", df2['Date'].max())

# Identify frequency
print("\ndf1 Frequency values:", df1['Frequency'].unique())
print("df2 Frequency values:", df2['Frequency'].unique())

# Standardize frequency strings
df1['Frequency'] = df1['Frequency'].str.strip()
df2['Frequency'] = df2['Frequency'].str.strip()

# Add Month and Year columns for easier grouping
df1['YearMonth'] = df1['Date'].dt.to_period('M')
df2['YearMonth'] = df2['Date'].dt.to_period('M')

# Time Series Analysis for df1 (Rural vs Urban)
df1_monthly = df1.groupby(['Date', 'Area'])['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df1_monthly, x='Date', y='Estimated Unemployment Rate (%)', hue='Area', marker='o')
plt.title('Monthly Unemployment Rate in India (Rural vs Urban: 2019-2020)')
plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', label='Covid-19 Lockdown Start')
plt.legend()
plt.grid(True)
plt.savefig('unemployment_rural_urban.png')

# Time Series Analysis for df2 (Regions)
df2_monthly = df2.groupby(['Date', 'Region.1'])['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df2_monthly, x='Date', y='Estimated Unemployment Rate (%)', hue='Region.1', marker='o')
plt.title('Monthly Unemployment Rate by Region (Jan-Oct 2020)')
plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', label='Covid-19 Lockdown Start')
plt.legend(title='Region')
plt.grid(True)
plt.savefig('unemployment_by_region.png')

# Calculate mean rate before and after March 2020 in df1
df1['Lockdown'] = df1['Date'] >= '2020-03-01'
impact_summary = df1.groupby(['Lockdown', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
print("Average Unemployment Rate Before vs After March 2020 (df1):")
print(impact_summary)

# Bar chart for Lockdown Impact
plt.figure(figsize=(10, 6))
impact_summary.plot(kind='bar', figsize=(10, 6))
plt.title('Impact of lockdown on Unemployment Rate')
plt.xticks([0, 1], ['Pre-Lockdown', 'Post-Lockdown'], rotation=0)
plt.ylabel('Average Unemployment Rate (%)')
plt.savefig('lockdown_impact_bar.png')

# Regional analysis using df2
# Top 10 States with highest average unemployment during the pandemic
state_impact = df2.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=state_impact.values, y=state_impact.index, palette='viridis')
plt.title('Average Unemployment Rate by State (Jan-Oct 2020)')
plt.xlabel('Unemployment Rate (%)')
plt.savefig('state_unemployment_ranking.png')

# Identify seasonal/key patterns:
# Let's check the peak month for all regions in df2
peak_month = df2.groupby('Date')['Estimated Unemployment Rate (%)'].mean().idxmax()
print(f"Peak Unemployment Month: {peak_month}")

# Seasonal check: Comparing same months in different datasets if possible
# df1 has May/June 2019 and May/June 2020.
may_june_2019 = df1[df1['Date'].dt.month.isin([5, 6]) & (df1['Date'].dt.year == 2019)]['Estimated Unemployment Rate (%)'].mean()
may_june_2020 = df1[df1['Date'].dt.month.isin([5, 6]) & (df1['Date'].dt.year == 2020)]['Estimated Unemployment Rate (%)'].mean()

print(f"Average Unemployment May-June 2019: {may_june_2019:.2f}%")
print(f"Average Unemployment May-June 2020: {may_june_2020:.2f}%")

# Get specific top 5 states
top_5_states = state_impact.head(5)
print("Top 5 States with highest unemployment in 2020:")
print(top_5_states)

# Region.1 analysis
region_impact = df2.groupby('Region.1')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
print("\nRegional average unemployment in 2020:")
print(region_impact)