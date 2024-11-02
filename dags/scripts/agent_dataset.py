import pandas as pd
import random

# Define lists for unique IDs and additional names for variety
num_rows = 100  # Adjust the number of rows as needed
names = [
    'Sophia Taylor', 'Jacob Brown', 'Mia Harris', 'Ethan Thomas', 'Amelia Thompson', 'Lucas White',
    'Olivia Moore', 'Mason Martinez', 'Isabella Anderson', 'Aiden Garcia', 'Emma Clark', 'Liam Rodriguez',
    'Benjamin Lewis', 'Ella Walker', 'James Hall', 'Avery Young', 'William King', 'Ava Wright', 'Elijah Scott',
    'Charlotte Green'
]

# Define the department-product mapping
department_product_mapping = {
    'Fraud and Security': ['Fraud Detection', 'Security Monitoring'],
    'Account Services': ['Savings Account', 'Current Account', 'Fixed Deposit'],
    'Customer Relations and Compliance': ['Customer Feedback', 'Compliance Check'],
    'Payments and Transactions': ['Credit Card', 'Debit Card', 'Bank Transfer'],
    'Loans and Credit': ['Personal Loan', 'Home Loan', 'Credit Card Loan']
}

# Define language and availability options
languages = ['English', 'Hindi', 'Spanish', 'French']
availability_options = [True, False]

# Generate the dataset
data = []
for i in range(1, num_rows + 1):
    name = random.choice(names)
    department = random.choice(list(department_product_mapping.keys()))
    product = random.choice(department_product_mapping[department])
    language = random.choice(languages)
    availability = random.choice(availability_options)
    data.append([i, name, language, department, product, availability])

# Create a DataFrame
columns = ['ID', 'Name', 'Language', 'Department', 'Product', 'Availability']
df = pd.DataFrame(data, columns=columns)

# Save to a CSV or Excel if needed
output_path = '/path/to/output/Agent_Dataset.csv'  # Specify your desired path
df.to_csv(output_path, index=False)
