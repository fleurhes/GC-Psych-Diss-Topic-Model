'''
This code opens the topic score CSV and analyzes how the topic probabilities changed over time.
'''

import pandas as pd
import matplotlib.pyplot as plt


csv_file = 'topic_scores.csv'
NUM_TOPICS = 10

df = pd.read_csv(csv_file)

# create a list of the topic column names
topic_columns = [f'Topic {i}' for i in range(NUM_TOPICS)]

# complete topic matrix
topic_matrix = df[topic_columns]

print("\n\ntopic matrix shape is ", topic_matrix.shape)
print(topic_matrix.head())

topic_labels = {
    0: 'Task, Memory, and Experience',
    1: 'Child learning and development',
    2: 'Experience and development',
    3: 'Identity and self-perception',
    4: 'College student Behaviors and Habits',
    5: 'Child parent relationships',
    6: 'Youth social development',
    7: 'Cognition and functionality in patients',
    8: 'Rat behavior and responses',
    9: 'Womens relationship with sexuality'
}

# overall topic distribution from the complete topic matrix
overall_topic_share = (
    topic_matrix
    .mean()
    * 100
)

topic_summary = pd.DataFrame({
    'Topic': range(NUM_TOPICS),
    'Label': [topic_labels[i] for i in range(NUM_TOPICS)],
    'Average Topic Share (%)': [
        overall_topic_share[f'Topic {i}']
        for i in range(NUM_TOPICS)
    ]
})

print("\n\noverall topic distribution:")
print(topic_summary)

overall_plot = pd.Series({
    topic_labels[i]:
    overall_topic_share[f'Topic {i}']
    for i in range(NUM_TOPICS)
})

overall_plot = overall_plot.sort_values()

overall_plot.plot(
    kind='barh',
    figsize=(10, 6)
)

plt.xlabel('Average Topic Share (%)')
plt.ylabel('')
plt.title('Overall Topic Distribution in Psychology Abstracts')
plt.tight_layout()
plt.show()

# compare the topic shares from the earliest and most recent decades
early = (
    df[
        df['year'].between(
            1980,
            1989
        )
    ][topic_columns]
    .mean()
    * 100
)

recent = (
    df[
        df['year'].between(
            2015,
            2024
        )
    ][topic_columns]
    .mean()
    * 100
)

change = recent - early

change.index = [
    topic_labels[i]
    for i in range(NUM_TOPICS)
]

change = change.sort_values()

print("\n\nchange from 1980-1989 to 2015-2024:")
print(change)

change.plot(
    kind='barh',
    figsize=(11, 7)
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    'Change in Average Topic Share (percentage points)'
)
plt.ylabel('')
plt.title(
    'How Psychology Abstract Topics Changed: '
    '1980–1989 vs. 2015–2024'
)
plt.tight_layout()
plt.show()

# calculate the average topic share for each year
yearly_topic_share = (
    df
    .groupby('year')[topic_columns]
    .mean()
    * 100
)

yearly_topic_share.columns = [
    topic_labels[i]
    for i in range(NUM_TOPICS)
]

# smooth the yearly averages using a 5-year rolling window
rolling_topic_share = (
    yearly_topic_share
    .rolling(
        window=5,
        center=True,
        min_periods=1
    )
    .mean()
)

# select the four topics with the largest changes
largest_changes = (
    change
    .abs()
    .sort_values(
        ascending=False
    )
    .head(4)
    .index
)

rolling_topic_share[
    largest_changes
].plot(
    figsize=(12, 6),
    linewidth=2
)

plt.xlabel('Year')
plt.ylabel('Average Topic Share (%)')
plt.title(
    'Psychology Research Topics With the Largest Changes Over Time'
)
plt.legend(
    title='Topic'
)
plt.tight_layout()
plt.show()

# save the summary data used in the analysis
topic_summary.to_csv(
    'psychology_topic_summary.csv',
    index=False
)

yearly_topic_share.to_csv(
    'psychology_yearly_topic_share.csv'
)

print("\nSaved:")
print("- topic_summary.csv")
print("-yearly_topic_share.csv")
