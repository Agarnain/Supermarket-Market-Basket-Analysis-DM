# Supermarket Market Basket Analysis

## Project Overview

This Python-based Data Mining mini project uses the **Apriori algorithm** and **Association Rule Mining** to discover products that are frequently purchased together in supermarket transactions.

The program generates a sample dataset of 10 supermarket transactions, converts it into a one-hot encoded DataFrame, finds frequent itemsets, and generates association rules.

## Objectives

- Analyze sample supermarket transactions.
- Find frequently purchased products and product combinations.
- Generate association rules between products.
- Interpret Support, Confidence, and Lift values.

## Technologies Used

- Python
- Pandas
- Mlxtend
- Apriori algorithm
- Association Rule Mining

## Project Files

```text
Supermarket-Market-Basket-Analysis/
├── agarnain_mini_project_DM.py
├── README.md
└── requirements.txt
```

## Dataset

The dataset is created inside the Python program; an external CSV file is not required.

It contains 10 sample transactions involving these products:

- Milk
- Bread
- Butter
- Eggs
- Beer
- Diapers

Each transaction is a list of products purchased together.

## How the Project Works

1. **Load transactions:** Generate the sample supermarket transactions.
2. **Preprocess data:** Use `TransactionEncoder` to convert transactions into a one-hot encoded Boolean DataFrame.
3. **Find frequent itemsets:** Apply Apriori with a minimum support of `0.30` (30%).
4. **Generate association rules:** Generate rules using confidence with a minimum threshold of `0.60` (60%).
5. **Rank the rules:** Sort the generated rules by Lift in descending order and display the strongest rule.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| Support | The proportion of all transactions containing an itemset. |
| Confidence | How often the consequent appears in transactions that contain the antecedent. |
| Lift | How much more often the antecedent and consequent occur together than expected if they were independent. |

## Installation and Execution

### 1. Clone the repository

Replace `your-username` with your GitHub username.

```bash
git clone https://github.com/your-username/Supermarket-Market-Basket-Analysis.git
```

### 2. Open the project folder

```bash
cd Supermarket-Market-Basket-Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
python agarnain_mini_project_DM.py
```

## Expected Output

When the program runs, it displays:

- The total number of sample transactions and a few examples.
- The one-hot encoded transaction matrix.
- Frequent itemsets and their support values.
- Association rules with Support, Confidence, and Lift.
- The strongest generated rule, including its confidence and lift values.

The exact rules depend on the sample transactions and the selected thresholds.

## Applications

- Understanding customer purchasing patterns.
- Supporting product placement decisions.
- Identifying products that are often purchased together.
- Supporting cross-selling and product recommendations.

## Future Scope

- Analyze a larger real-world supermarket dataset.
- Add charts to visualize frequent itemsets and association rules.
- Build an interactive dashboard.
- Explore other association-rule mining methods.

## Author

**Agarnain**

## Purpose

This project is developed for educational purposes as a Data Mining mini project.
