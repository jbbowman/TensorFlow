import numpy as np
import pandas as pd

class Customer:
    instance_counter = 0

    def __init__(self):
        self.id = Customer.instance_counter
        self.credit_score = (np.random.randint(300, 851) - 300) / 550
        self.debt_income_ratio = np.random.random()
        self.loan_value_ratio = np.random.random()
        self.age = np.random.randint(18, 75) / 75
        self.default = np.random.rand() < 0.5 * self.debt_income_ratio + 0.5 * self.loan_value_ratio - 0.5 * self.credit_score
        Customer.instance_counter += 1

    def __str__(self):
        return self.id

def generate_customers(num_customers):
    customer_data = []
    for i in range(num_customers):
        customer = Customer()
        customer_data.append([customer.credit_score, customer.debt_income_ratio, customer.loan_value_ratio, customer.age, customer.default])
    return customer_data

def generate_dataframe(customer_data):
    return pd.DataFrame(customer_data, columns=["Credit Score", "Debt/Income", "Loan/Value", "Age", "Default"])

def prep_data(data):
    rg = np.random.default_rng()

    features = data.drop(columns='Default').to_numpy()
    targets = data['Default'].to_numpy()
    weights = rg.random(features.shape[1])
    return features, targets, weights

if __name__ == '__main__':
    customer_data = generate_customers(5)
    data = generate_dataframe(customer_data)
    prepped_data = prep_data(data)

    print(customer_data, data, prepped_data, sep="\n\n")
