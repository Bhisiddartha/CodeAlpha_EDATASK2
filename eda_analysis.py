import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import os

# ============================================================
# CODEALPHA TASK 2 - EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 70)
print("        CODEALPHA TASK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# PART 1 - LOAD DATASET
# ============================================================

file_path = "dataset/superstore.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nColumn Names:")
for i, column in enumerate(df.columns, 1):
    print(i, ".", column)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nFirst 5 Records:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

print("\nMissing Value Percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))


# ============================================================
# DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATE RECORD ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("\nDuplicate Records:", duplicate_count)

print("Unique Records:", df.drop_duplicates().shape[0])


# ============================================================
# NUMERICAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("4. NUMERICAL VARIABLES")
print("=" * 70)

numerical_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical Columns:")

for column in numerical_columns:
    print("-", column)


# ============================================================
# CATEGORICAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("5. CATEGORICAL VARIABLES")
print("=" * 70)

categorical_columns = [
    "Order ID",
    "Order Date",
    "Customer Name",
    "Segment",
    "Country",
    "City",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name"
]

print("\nCategorical Columns:")

for column in categorical_columns:
    print("-", column)


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. UNIQUE VALUES")
print("=" * 70)

for column in categorical_columns:
    print(column, ":", df[column].nunique())


# ============================================================
# NEGATIVE VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. NEGATIVE VALUE CHECK")
print("=" * 70)

for column in numerical_columns:
    negative_count = (df[column] < 0).sum()
    print(column, "->", negative_count, "negative values")


# ============================================================
# PART 2 - DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - DATA CLEANING")
print("=" * 70)

cleaned_df = df.copy()

# Fill missing Customer Name
cleaned_df["Customer Name"] = cleaned_df[
    "Customer Name"
].fillna("Unknown Customer")

# Fill missing City
cleaned_df["City"] = cleaned_df[
    "City"
].fillna("Unknown City")

# Convert Order Date to datetime
cleaned_df["Order Date"] = pd.to_datetime(
    cleaned_df["Order Date"]
)

# Remove duplicate records
records_before = len(cleaned_df)

cleaned_df = cleaned_df.drop_duplicates()

records_after = len(cleaned_df)

print("\nMissing values handled successfully.")

print("Duplicate records removed:",
      records_before - records_after)

print("Records before cleaning:",
      records_before)

print("Records after cleaning:",
      records_after)

print("\nRemaining Missing Values:")
print(cleaned_df.isnull().sum())

print("\nData cleaning completed!")


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

cleaned_df.to_csv(
    "dataset/cleaned_superstore.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("dataset/cleaned_superstore.csv")


# ============================================================
# PART 3 - TREND AND PATTERN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - TREND AND PATTERN ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. SALES BY CATEGORY
# ------------------------------------------------------------

category_sales = cleaned_df.groupby(
    "Category"
)["Sales"].sum().sort_values(ascending=False)

print("\n1. SALES BY CATEGORY")
print(category_sales)


# ------------------------------------------------------------
# 2. SALES BY REGION
# ------------------------------------------------------------

region_sales = cleaned_df.groupby(
    "Region"
)["Sales"].sum().sort_values(ascending=False)

print("\n2. SALES BY REGION")
print(region_sales)


# ------------------------------------------------------------
# 3. PROFIT BY CATEGORY
# ------------------------------------------------------------

category_profit = cleaned_df.groupby(
    "Category"
)["Profit"].sum().sort_values(ascending=False)

print("\n3. PROFIT BY CATEGORY")
print(category_profit)


# ------------------------------------------------------------
# 4. SALES BY CUSTOMER SEGMENT
# ------------------------------------------------------------

segment_sales = cleaned_df.groupby(
    "Segment"
)["Sales"].sum().sort_values(ascending=False)

print("\n4. SALES BY CUSTOMER SEGMENT")
print(segment_sales)


# ------------------------------------------------------------
# 5. TOP 10 PRODUCTS
# ------------------------------------------------------------

top_products = cleaned_df.groupby(
    "Product Name"
)["Sales"].sum().sort_values(
    ascending=False
).head(10)

print("\n5. TOP 10 PRODUCTS BY SALES")
print(top_products)


# ------------------------------------------------------------
# 6. MONTHLY SALES TREND
# ------------------------------------------------------------

cleaned_df["Year-Month"] = cleaned_df[
    "Order Date"
].dt.to_period("M")

monthly_sales = cleaned_df.groupby(
    "Year-Month"
)["Sales"].sum()

print("\n6. MONTHLY SALES TREND")
print(monthly_sales)


# ------------------------------------------------------------
# 7. AVERAGE PROFIT BY DISCOUNT
# ------------------------------------------------------------

discount_profit = cleaned_df.groupby(
    "Discount"
)["Profit"].mean()

print("\n7. AVERAGE PROFIT BY DISCOUNT")
print(discount_profit)

print("\nTrend and pattern analysis completed!")


# ============================================================
# PART 4 - OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - OUTLIER DETECTION")
print("=" * 70)


def find_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    return outliers, lower_limit, upper_limit


# Sales outliers
sales_outliers, sales_lower, sales_upper = find_outliers(
    cleaned_df,
    "Sales"
)

print("\nSales Outliers:", len(sales_outliers))
print("Sales Lower Limit:", round(sales_lower, 2))
print("Sales Upper Limit:", round(sales_upper, 2))


# Profit outliers
profit_outliers, profit_lower, profit_upper = find_outliers(
    cleaned_df,
    "Profit"
)

print("\nProfit Outliers:", len(profit_outliers))
print("Profit Lower Limit:", round(profit_lower, 2))
print("Profit Upper Limit:", round(profit_upper, 2))


# ============================================================
# PART 5 - CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 5 - CORRELATION ANALYSIS")
print("=" * 70)

correlation = cleaned_df[
    ["Sales", "Quantity", "Discount", "Profit"]
].corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))


# ============================================================
# PART 6 - STATISTICAL HYPOTHESIS TEST
# ============================================================

print("\n" + "=" * 70)
print("PART 6 - STATISTICAL HYPOTHESIS TEST")
print("=" * 70)

print("\nQuestion:")
print("Does profit differ significantly between")
print("orders with and without discount?")


no_discount_profit = cleaned_df[
    cleaned_df["Discount"] == 0
]["Profit"]

discounted_profit = cleaned_df[
    cleaned_df["Discount"] > 0
]["Profit"]


t_statistic, p_value = ttest_ind(
    no_discount_profit,
    discounted_profit,
    equal_var=False
)

print("\nT-statistic:", round(t_statistic, 4))
print("P-value:", round(p_value, 6))


if p_value < 0.05:

    print(
        "Result: There is a statistically significant "
        "difference in average profit."
    )

else:

    print(
        "Result: There is no statistically significant "
        "difference in average profit."
    )


# ============================================================
# PART 7 - LOSS-MAKING ORDERS
# ============================================================

print("\n" + "=" * 70)
print("PART 7 - LOSS ANALYSIS")
print("=" * 70)

loss_orders = cleaned_df[
    cleaned_df["Profit"] < 0
]

loss_count = len(loss_orders)

loss_percentage = (
    loss_count / len(cleaned_df)
) * 100

print("\nLoss-making orders:", loss_count)

print(
    "Percentage of loss-making orders:",
    round(loss_percentage, 2),
    "%"
)


# ============================================================
# PART 8 - DATA VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("PART 8 - DATA VISUALIZATION")
print("=" * 70)


# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# GRAPH 1 - SALES BY CATEGORY
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

category_sales.plot(
    kind="bar"
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/01_sales_by_category.png",
    dpi=300
)

plt.close()

print("Saved: 01_sales_by_category.png")


# ------------------------------------------------------------
# GRAPH 2 - SALES BY REGION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

region_sales.plot(
    kind="bar"
)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/02_sales_by_region.png",
    dpi=300
)

plt.close()

print("Saved: 02_sales_by_region.png")


# ------------------------------------------------------------
# GRAPH 3 - MONTHLY SALES TREND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "visualizations/03_monthly_sales_trend.png",
    dpi=300
)

plt.close()

print("Saved: 03_monthly_sales_trend.png")


# ------------------------------------------------------------
# GRAPH 4 - DISCOUNT VS PROFIT
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    discount_profit.index,
    discount_profit.values,
    marker="o"
)

plt.title("Average Profit by Discount")
plt.xlabel("Discount")
plt.ylabel("Average Profit")

plt.tight_layout()

plt.savefig(
    "visualizations/04_discount_vs_profit.png",
    dpi=300
)

plt.close()

print("Saved: 04_discount_vs_profit.png")


# ------------------------------------------------------------
# GRAPH 5 - CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "visualizations/05_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Saved: 05_correlation_heatmap.png")


# ============================================================
# PART 9 - FINAL INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("PART 9 - FINAL INSIGHTS")
print("=" * 70)


highest_sales_category = category_sales.index[0]
highest_sales_region = region_sales.index[0]
highest_profit_category = category_profit.index[0]
highest_segment = segment_sales.index[0]
top_product = top_products.index[0]

highest_month = monthly_sales.idxmax()

print("\n1. Highest Sales Category:")
print(
    highest_sales_category,
    "with",
    round(category_sales.iloc[0], 2)
)

print("\n2. Highest Sales Region:")
print(
    highest_sales_region,
    "with",
    round(region_sales.iloc[0], 2)
)

print("\n3. Highest Profit Category:")
print(
    highest_profit_category,
    "with",
    round(category_profit.iloc[0], 2)
)

print("\n4. Highest Sales Customer Segment:")
print(
    highest_segment,
    "with",
    round(segment_sales.iloc[0], 2)
)

print("\n5. Top Selling Product:")
print(
    top_product,
    "with",
    round(top_products.iloc[0], 2)
)

print("\n6. Highest Sales Month:")
print(
    highest_month,
    "with",
    round(monthly_sales.max(), 2)
)

print("\n7. Loss-making Orders:")
print(
    loss_count,
    "orders",
    "(",
    round(loss_percentage, 2),
    "%)"
)

print("\n8. Discount and Profit:")
print(
    "Average profit generally decreases as discount increases."
)

print("\n9. Data Quality:")
print(
    "Missing values were handled and duplicate records were removed."
)

print("\n10. Outliers:")
print(
    "Sales and Profit outliers were identified using the IQR method."
)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("        CODEALPHA TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nFiles generated:")
print("- dataset/cleaned_superstore.csv")
print("- visualizations/01_sales_by_category.png")
print("- visualizations/02_sales_by_region.png")
print("- visualizations/03_monthly_sales_trend.png")
print("- visualizations/04_discount_vs_profit.png")
print("- visualizations/05_correlation_heatmap.png")

print("\nEDA project completed!")