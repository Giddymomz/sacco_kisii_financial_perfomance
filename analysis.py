# Analysis for SACCO Research Proposal
# Gideon Mogendi Momanyi - Kisii County, Kenya (2027)
# Model: ROA = b0 + b1*Credit + b2*Capital + b3*Liquidity + b4*Portfolio + e

import pandas as pd
import statsmodels.api as sm

# 1. Load data
df = pd.read_csv("sample_data.csv")
print("Data loaded:")
print(df.head())
print()
print(df.describe())
print()

# 2. Define variables
# X = Independent variables (investment decisions)
# Y = Dependent variable (financial performance - ROA)
X = df[["credit_investment", "capital_investment", "liquidity_investment", "portfolio_investment"]]
y = df["roa"]

# 3. Add constant (b0)
X = sm.add_constant(X)

# 4. Run Multiple Linear Regression
model = sm.OLS(y, X).fit()

# 5. Print results
print("=== REGRESSION RESULTS ===")
print(model.summary())

# 6. Correlation
print()
print("=== CORRELATION WITH ROA ===")
print(df.corr(numeric_only=True)["roa"].sort_values(ascending=False))

print()
print("Done. Check p-values < 0.05 for significant effects.")
