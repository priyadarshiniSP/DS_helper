import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from column_detector import detect_column_types

def visualize(df: pd.DataFrame):
    """
    Automatically generates appropriate plots based on detected column types.
    """
    types = detect_column_types(df)

    # Numerical columns
    for col in types['numerical']:
        # Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

        # Boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        plt.ylabel(col)
        plt.show()

    # Scatter plot if at least two numerical columns
    if len(types['numerical']) >= 2:
        col1, col2 = types['numerical'][0], types['numerical'][1]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[col1].dropna(), y=df[col2].dropna())
        plt.title(f'Scatter plot of {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

    # Categorical columns
    for col in types['categorical']:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[col].dropna())
        plt.title(f'Count plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    # Text columns
    for col in types['text']:
        # Combine all text in the column
        text = ' '.join(df[col].dropna().astype(str))

        # Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud of {col}')
        plt.show()

        # Frequency plot (top 20 words)
        words = text.split()
        freq = Counter(words).most_common(20)
        if freq:
            words_list, counts = zip(*freq)
            plt.figure(figsize=(10, 6))
            plt.bar(words_list, counts)
            plt.title(f'Top 20 Words in {col}')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.show()
