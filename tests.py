
import pandas as pd
from ds_helper.column_detector import detect_column_types
from ds_helper.auto_visualizer import plot_numerical, plot_categorical, plot_pairplot, plot_text
from ds_helper.text_cleaner import TextCleaner

# Load the Titanic dataset
df = pd.read_csv(r"C:\Users\PriyaBackup\Downloads\Titanic-Dataset.csv")


# Detect column types
print("Column Types:")
column_types = detect_column_types(df)
print(column_types)


# Visualize the data
print("\nVisualizing Data:")
if column_types['numerical']:
    plot_numerical(df, column_types['numerical'])
    plot_pairplot(df, column_types['numerical'])
if column_types['categorical']:
    plot_categorical(df, column_types['categorical'])
if column_types['text']:
    plot_text(df, column_types['text'])

# Try out text cleaning on the 'Name' column if it exists
if 'Name' in df.columns:
    cleaner = TextCleaner()
    sample_name = df['Name'].iloc[0]
    print("\nOriginal Name:", sample_name)
    print("Cleaned Name:", cleaner.clean(str(sample_name)))