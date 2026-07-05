import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/online_retail_cleaned.csv")

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

# Create snapshot date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Group by customer
customer_groups = df.groupby("CustomerID")

# Frequency
frequency = customer_groups["InvoiceNo"].nunique()

# Monetary
monetary = customer_groups["Revenue"].sum()

# Last purchase
last_purchase = customer_groups["InvoiceDate"].max()

# Recency
recency = snapshot_date - last_purchase
recency = recency.dt.days

# Create RFM DataFrame
rfm = pd.DataFrame({
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary
})

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    q=5,
    labels=[5,4,3,2,1]
)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    q=5,
    labels=[1,2,3,4,5]
)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    q=5,
    labels=[1,2,3,4,5]
)
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

def customer_segment(score):

    if score == "555":
        return "Champion"

    elif score.startswith("55"):
        return "Loyal Customer"

    elif score.startswith("5"):
        return "Potential Loyalist"

    elif score.startswith("1"):
        return "Lost Customer"

    else:
        return "Others"
    
rfm["Segment"] = rfm["RFM_Score"].apply(customer_segment)
rfm = rfm.reset_index()

segment_summary = rfm.groupby("Segment").agg({
    "CustomerID": "count",
    "Monetary": "mean",
    "Frequency": "mean",
    "Recency": "mean"
})

segment_summary = segment_summary.sort_values(
    by="CustomerID",
    ascending=False
)

print(segment_summary)
segment_summary.to_csv(
    "Dashboard/segment_summary.csv"
)
rfm.to_csv("Dashboard/rfm_customers.csv", index=False)