'''
This code prepares the psychology-related dissertation abstracts for the topic model.
'''

import pandas as pd
import re
import matplotlib.pyplot as plt


def prepare_psychology_data(data, headline_column):
    # inspect the full dataset before filtering
    print("\n\ndataset shape is ", data.shape)
    print(data[['year', 'program', 'program_coarse', headline_column]].head())

    print("\n\nall records by year:")
    print(data['year'].value_counts().sort_index())

    # keep psychology-related programs
    PSYCHOLOGY_PATTERN = 'psycholog'

    program_match = (
        data['program']
        .fillna('')
        .str.contains(
            PSYCHOLOGY_PATTERN,
            case=False,
            regex=False
        )
    )

    coarse_match = (
        data['program_coarse']
        .fillna('')
        .str.contains(
            PSYCHOLOGY_PATTERN,
            case=False,
            regex=False
        )
    )

    psychology_match = (
        program_match |
        coarse_match
    )

    print(
        "\nRecords where program contains psychology:",
        program_match.sum()
    )

    print(
        "Records where program_coarse contains psychology:",
        coarse_match.sum()
    )

    print(
        "Unique records included by either field:",
        psychology_match.sum()
    )

    print("\nIncluded normalized program labels:")
    print(
        data.loc[
            psychology_match,
            'program'
        ]
        .value_counts()
    )

    # audit program_raw without using it as the inclusion rule
    raw_psychology_match = (
        data['program_raw']
        .fillna('')
        .str.contains(
            PSYCHOLOGY_PATTERN,
            case=False,
            regex=False
        )
    )

    raw_only = (
        raw_psychology_match &
        ~psychology_match
    )

    print(
        "\nRecords where only program_raw mentions psychology:",
        raw_only.sum()
    )

    print("\nExamples of raw-only matches:")
    print(
        data.loc[
            raw_only,
            ['program', 'program_coarse', 'program_raw']
        ]
        .head(15)
    )

    # remove unusable and duplicate records
    psychology = data.loc[psychology_match].copy()

    psychology['year'] = pd.to_numeric(
        psychology['year'],
        errors='coerce'
    )

    psychology = psychology.dropna(
        subset=[headline_column, 'year']
    )

    psychology = psychology[
        psychology['abstract_is_placeholder'] == False
    ].copy()

    psychology['dedupe_key'] = (
        psychology['dup_group']
        .fillna(psychology['record_id'])
    )

    psychology = (
        psychology
        .sort_values(
            'abstract_wordcount',
            ascending=False
        )
        .drop_duplicates(
            subset='dedupe_key',
            keep='first'
        )
        .sort_values('year')
        .reset_index(drop=True)
    )

    print(
        "\nUsable unique psychology-related dissertations:",
        len(psychology)
    )

    print(
        "Full usable year range:",
        int(psychology['year'].min()),
        'to',
        int(psychology['year'].max())
    )

    print("\nUsable psychology-related records by year before the final year filter:")
    print(
        psychology['year']
        .value_counts()
        .sort_index()
        .to_string()
    )

    # final analysis period
    START_YEAR = 1980
    END_YEAR = 2024

    psychology = psychology[
        (psychology['year'] >= START_YEAR) &
        (psychology['year'] <= END_YEAR)
    ].copy()

    psychology = psychology.reset_index(drop=True)

    print(
        "\nPsychology-related dissertations used in analysis:",
        len(psychology)
    )

    print(
        "Analysis years:",
        int(psychology['year'].min()),
        'to',
        int(psychology['year'].max())
    )

    year_counts = (
        psychology['year']
        .value_counts()
        .sort_index()
    )

    year_counts.plot(
        kind='bar',
        figsize=(14, 5)
    )

    plt.xlabel('Year')
    plt.ylabel('Number of Usable Abstracts')
    plt.title('Usable Psychology-Related Dissertations by Year')
    plt.tight_layout()
    plt.show()

    # remove known legacy formatting from the abstract text
    def clean_legacy_formatting(text):
        if pd.isna(text) or not isinstance(text, str):
            return text

        text = re.sub(
            r'\\?\{(?:dollar|lcub|rcub)\}',
            ' ',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'\\(?:sp|sb)\d*',
            ' ',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'\b(?:dollar|lcub|rcub)\b',
            ' ',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'\(DIAGRAM, TABLE OR GRAPHIC OMITTED[^\)]*DAI\)',
            ' ',
            text,
            flags=re.IGNORECASE
        )

        text = text.replace('\\', ' ')
        text = text.replace('{', ' ')
        text = text.replace('}', ' ')

        return text

    artifact_pattern = r'\{dollar\}|\{lcub\}|\{rcub\}|\\sp|\\sb'

    artifact_mask = psychology[headline_column].str.contains(
        artifact_pattern,
        case=False,
        regex=True,
        na=False
    )

    print(
        "\nPsychology abstracts containing known legacy formatting:",
        artifact_mask.sum()
    )

    psychology[headline_column] = (
        psychology[headline_column]
        .map(clean_legacy_formatting)
    )

    # save the exact abstract corpus used by the model
    psychology[
        ['record_id', 'year', 'program', headline_column]
    ].to_csv(
        'filtered_corpus.csv',
        index=False
    )

    return psychology
