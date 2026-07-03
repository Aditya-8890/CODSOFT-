import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

df = pd.read_csv('Titanic-Dataset (1).csv')
os.makedirs('plots', exist_ok=True)
df.info()   
print(df.head(10))    
plt.figure(figsize=(6, 5))
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None, palette='muted', hue='Sex', legend=False)
plt.title('Survival Rate by Gender')
plt.savefig('plots/survival_by_sex.png')
plt.show()

plt.figure(figsize=(6, 5))
sns.barplot(x='Pclass', y='Survived', data=df, errorbar=None, palette='viridis', hue='Pclass', legend=False)
plt.title('Survival Rate by Passenger Class')
plt.savefig('plots/survival_by_pclass.png')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, multiple='stack', palette='crest', bins=30)
plt.title('Age Distribution by Survival Status')
plt.savefig('plots/age_distribution.png')
plt.show()

plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.savefig('plots/correlation_heatmap.png')
plt.show()

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['HasCabin'] = df['Cabin'].notnull().astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df = pd.get_dummies(df, columns=['Pclass', 'Embarked', 'Title'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.legend()
plt.savefig('plots/roc_curve.png')
plt.show()

coefficients = model.coef_[0]
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coeff_df['Abs'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='coolwarm', hue='Feature', legend=False)
plt.title('Logistic Regression Coefficients')
plt.savefig('plots/feature_coefficients.png')
plt.show()
