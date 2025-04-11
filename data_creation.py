import pandas as pd

# Define the data as a list of tuples
data = [
    ("Dushyant is a data scientist focused on AI agents.", "10000000000000000001"),
    ("Sridhar is 25 and works in fintech.", "10000000000000000002")
]

# Create DataFrame with column names
df = pd.DataFrame(data, columns=["text", "person_id"])

# Display the DataFrame
print(df)

# Save to CSV
df.to_csv("user_data_temp.csv", index=False)