import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
from ds_helper.column_detector import detect_column_types

df = pd.DataFrame({
    'age': [23, 45, 36, 27, 50, 29],
    'salary': [50000, 80000, 60000, 52000, 90000, 58000],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],
    'comments': [
        'Great team and friendly environment',
        'Need better communication from management',
        'Loved the flexible hours',
        'The workload is heavy',
        'Supportive leadership',
        'Better work-life balance needed'

    ]
})

def plot_numerical(df, numerical_cols):
    for col in numerical_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
        plt.ylabel(col)
        plt.show()

def plot_pairplot(df, numerical_cols):
    sns.pairplot(df[numerical_cols])
    plt.suptitle("Pairplot of Numerical Columns", y=1.02)
    plt.show()

def plot_categorical(df, categorical_cols):
    for col in categorical_cols:
        sns.countplot(x=df[col])
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

def plot_text(df, text_cols):
    for col in text_cols:
        text_data = ' '.join(df[col].dropna().astype(str)).lower()
        text_data = re.sub(r'[^\w\s]', '', text_data)
        words = text_data.split()
        words = [word for word in words if word not in STOPWORDS]

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud of {col}')
        plt.show()

        freq = Counter(words).most_common(20)
        words_list, counts = zip(*freq)
        plt.figure(figsize=(10, 6))
        plt.bar(words_list, counts)
        plt.title(f'Top 20 Words in {col}')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()

column_types = detect_column_types(df)

if column_types['numerical']:
    plot_numerical(df, column_types['numerical'])
    plot_pairplot(df, column_types['numerical'])
if column_types['categorical']:
    plot_categorical(df, column_types['categorical'])
if column_types['text']:
    plot_text(df, column_types['text'])
