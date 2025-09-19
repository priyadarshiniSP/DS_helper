import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re

# Sample DataFrame
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

# Visualize numerical columns
def plot_numerical(df):
    numerical_cols = ['age', 'salary']
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

# Pairplot
def plot_pairplot(df):
    sns.pairplot(df[['age', 'salary']])
    plt.suptitle("Pairplot of Numerical Columns", y=1.02)
    plt.show()

# Visualize categorical column
def plot_categorical(df):
    sns.countplot(x=df['gender'])
    plt.title('Count Plot of gender')
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

# Visualize text column
def plot_text(df):
    text_data = ' '.join(df['comments'].dropna().astype(str)).lower()
    text_data = re.sub(r'[^\w\s]', '', text_data)
    words = text_data.split()
    words = [word for word in words if word not in STOPWORDS]

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of comments')
    plt.show()

    # Frequency Bar Chart
    freq = Counter(words).most_common(20)
    words_list, counts = zip(*freq)
    plt.figure(figsize=(10, 6))
    plt.bar(words_list, counts)
    plt.title('Top 20 Words in comments')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Run all visualizations
plot_numerical(df)
plot_pairplot(df)
plot_categorical(df)
plot_text(df)
