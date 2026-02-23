import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# -----------------------------
# STEP 1: Create Sample Dataset
# -----------------------------

np.random.seed(42)

data = {
    "Order Date": pd.date_range(start="2023-01-01", periods=200, freq="D"),
    "Product Name": np.random.choice(["Laptop", "Mobile", "Tablet", "Printer", "Headphones"], 200),
    "Category": np.random.choice(["Electronics", "Accessories"], 200),
    "Region": np.random.choice(["North", "South", "East", "West"], 200),
    "Sales": np.random.randint(1000, 50000, 200),
    "Quantity": np.random.randint(1, 10, 200),
    "Profit": np.random.randint(100, 10000, 200)
}

df = pd.DataFrame(data)

print("Sample Data:")
print(df.head())

# -----------------------------
# STEP 2: Feature Engineering
# -----------------------------

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# -----------------------------
# STEP 3: Revenue Trend
# -----------------------------

monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure()
monthly_sales.plot()
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# -----------------------------
# STEP 4: Top Selling Products
# -----------------------------

top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False)

plt.figure()
top_products.plot(kind="bar")
plt.title("Top Selling Products")
plt.ylabel("Total Sales")
plt.show()

# -----------------------------
# STEP 5: Category Performance
# -----------------------------

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure()
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.show()

# -----------------------------
# STEP 6: Regional Performance
# -----------------------------

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure()
region_sales.plot(kind="bar")
plt.title("Regional Sales Performance")
plt.ylabel("Total Sales")
plt.show()

# -----------------------------
# STEP 7: Data Scaling
# -----------------------------

scaler = MinMaxScaler()
df[["Sales", "Profit"]] = scaler.fit_transform(df[["Sales", "Profit"]])

print("\nScaled Data Sample:")
print(df.head())

# -----------------------------
# STEP 8: Simple Prediction
# -----------------------------

X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("\nPrediction Sample:")
print(predictions[:5])