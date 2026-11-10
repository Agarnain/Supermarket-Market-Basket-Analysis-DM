import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def generate_transaction_data():
    """Generates a sample dataset of supermarket customer transactions."""
    transactions = [
        ['Milk', 'Bread', 'Butter', 'Eggs'],
        ['Beer', 'Bread', 'Diapers'],
        ['Milk', 'Beer', 'Bread', 'Diapers'],
        ['Milk', 'Butter', 'Eggs'],
        ['Bread', 'Butter', 'Eggs'],
        ['Milk', 'Bread', 'Butter', 'Beer', 'Diapers'],
        ['Milk', 'Bread', 'Butter'],
        ['Beer', 'Diapers'],
        ['Milk', 'Bread', 'Diapers'],
        ['Milk', 'Butter', 'Eggs', 'Beer']
    ]
    return transactions

def main():
    print("--- 1. Loading Transaction Data ---")
    dataset = generate_transaction_data()
    print(f"Total Transactions Sampled: {len(dataset)}")
    for i, trans in enumerate(dataset[:3]):
        print(f"Transaction {i+1}: {trans}")
    print("...")

    print("\n--- 2. Data Preprocessing & One-Hot Encoding ---")
    # TransactionEncoder transforms the list of lists into a one-hot encoded boolean DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("One-Hot Encoded Transaction Matrix (First 5 rows):")
    print(df.head())

    print("\n--- 3. Mining Frequent Itemsets using Apriori ---")
    # Find itemsets with a minimum support of 30% (0.3)
    frequent_itemsets = apriori(df, min_support=0.3, use_column_names=True)
    
    print("Frequent Itemsets Found:")
    print(frequent_itemsets.sort_values(by="support", ascending=False))

    print("\n--- 4. Generating Association Rules ---")
    # Generate rules based on Confidence metric (e.g., minimum confidence of 60%)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    
    # Sort rules by 'lift' to find the strongest associations
    rules = rules.sort_values(by="lift", ascending=False)
    
    # Display key columns of the discovered rules
    print("Top Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

    print("\n--- 5. Practical Example Insight ---")
    if not rules.empty:
        top_rule = rules.iloc[0]
        print(f"Strongest Rule Found:")
        print(f"If a customer buys: {list(top_rule['antecedents'])}")
        print(f"They are likely to buy: {list(top_rule['consequents'])}")
        print(f"Confidence Level: {top_rule['confidence']*100:.1f}%")
        print(f"Lift Value: {top_rule['lift']:.2f}")

if __name__ == "__main__":
    main()