STEP 1: IMPORTING LIBRERIES & LOAD DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


STEP 2: UNDERSTANDING THE DATA
# Shape and preview
print("Shape of dataset:", df.shape)
df.head()

# Info and summary
df.info()
df.describe()

# Null values
df.isnull().sum()

# Unique values
df.nunique()


STEP 3: UNIVARIATE ANALYSIS
NUMERICAL COLUMNS
# Age
sns.histplot(df['Age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

# Fare
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

CATEGORICAL COLUMNS
# Sex
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# Pclass
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()


STEP 4: BIVARIATE ANALYSIS
# Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.show()

# Survival by Class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival Count by Passenger Class")
plt.show()


STEP 5: MULTIVARIATE ANALYSIS
# Pairplot
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


STEP 6: MISSING VALUE HANDLING
# Pairplot
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived')
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


STEP 7: SKEWNESS CHECK
# Check skew
df[['Age', 'Fare']].skew()

# Log transformation on Fare if highly skewed
df['Fare_log'] = np.log1p(df['Fare'])

sns.histplot(df['Fare_log'], kde=True)
plt.title("Log Transformed Fare")
plt.show()


STEP 8: SUMMARY OF INSIGHTS
### Summary of EDA Findings:
- Majority of passengers were in 3rd class and male.
- Females had a significantly higher survival rate.
- Younger passengers and higher fare payers had higher survival chances.
- Age and Fare distributions had outliers; Fare was skewed.
- Pclass and Sex showed strong relationships with survival.
