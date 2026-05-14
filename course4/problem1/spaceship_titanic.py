import pandas as pd
import matplotlib.pyplot as plt


def load_and_merge_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    merged_df = pd.concat([train_df, test_df], ignore_index=True)

    return train_df, test_df, merged_df


def print_data_count(train_df, test_df, merged_df):
    print('=== 데이터 수량 ===')
    print(f'train.csv 행 수: {len(train_df)}')
    print(f'test.csv 행 수: {len(test_df)}')
    print(f'병합 후 전체 행 수: {len(merged_df)}')
    print(f'전체 열 수: {len(merged_df.columns)}')
    print()


def preprocess_transported(df):
    transported_df = df[df['Transported'].notna()].copy()
    transported_df['Transported_num'] = transported_df['Transported'].astype(int)
    return transported_df


def calculate_numeric_relation(df, column_name):
    valid_df = df[[column_name, 'Transported_num']].dropna()

    if len(valid_df) < 2:
        return 0.0

    correlation = valid_df[column_name].corr(valid_df['Transported_num'])

    if pd.isna(correlation):
        return 0.0

    return abs(correlation)


def calculate_categorical_relation(df, column_name):
    valid_df = df[[column_name, 'Transported_num']].dropna()

    if len(valid_df) == 0:
        return 0.0

    grouped = valid_df.groupby(column_name)['Transported_num'].mean()
    overall_mean = valid_df['Transported_num'].mean()

    weighted_score = 0.0
    total_count = len(valid_df)

    for category, mean_value in grouped.items():
        category_count = len(valid_df[valid_df[column_name] == category])
        weighted_score += (
            category_count / total_count
        ) * ((mean_value - overall_mean) ** 2)

    return weighted_score


def find_most_related_feature(df):
    excluded_columns = ['PassengerId', 'Name', 'Transported', 'Transported_num']
    candidate_columns = [
        column for column in df.columns if column not in excluded_columns
    ]

    relation_scores = {}

    for column in candidate_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            score = calculate_numeric_relation(df, column)
        else:
            score = calculate_categorical_relation(df, column)

        relation_scores[column] = score

    sorted_scores = sorted(
        relation_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print('=== Transported와의 관련성 순위 ===')
    for feature, score in sorted_scores:
        print(f'{feature}: {score:.6f}')
    print()

    most_related_feature = sorted_scores[0]
    print(
        f'가장 관련성이 높은 항목: '
        f'{most_related_feature[0]} (점수: {most_related_feature[1]:.6f})'
    )
    print()


def make_age_group(age):
    if pd.isna(age):
        return 'Unknown'

    age = int(age)

    if 10 <= age < 20:
        return '10s'
    if 20 <= age < 30:
        return '20s'
    if 30 <= age < 40:
        return '30s'
    if 40 <= age < 50:
        return '40s'
    if 50 <= age < 60:
        return '50s'
    if 60 <= age < 70:
        return '60s'
    if 70 <= age < 80:
        return '70s'

    return 'Other'


def plot_transport_by_age_group(df):
    age_df = df[df['Age'].notna()].copy()
    age_df['AgeGroup'] = age_df['Age'].apply(make_age_group)

    ordered_groups = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']

    filtered_df = age_df[age_df['AgeGroup'].isin(ordered_groups)]

    summary = (
        filtered_df.groupby(['AgeGroup', 'Transported'])
        .size()
        .unstack(fill_value=0)
        .reindex(ordered_groups)
    )

    summary.plot(
        kind='bar',
        figsize=(10, 6)
    )

    plt.title('Transported Status by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Passengers')
    plt.xticks(rotation=0)
    plt.legend(title='Transported')
    plt.tight_layout()
    plt.show()


def plot_destination_age_distribution(df):
    bonus_df = df[df['Age'].notna() & df['Destination'].notna()].copy()
    bonus_df['AgeGroup'] = bonus_df['Age'].apply(make_age_group)

    ordered_groups = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']
    filtered_df = bonus_df[bonus_df['AgeGroup'].isin(ordered_groups)]

    summary = (
        filtered_df.groupby(['Destination', 'AgeGroup'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ordered_groups)
    )

    summary.plot(
        kind='bar',
        figsize=(12, 7)
    )

    plt.title('Age Group Distribution by Destination')
    plt.xlabel('Destination')
    plt.ylabel('Number of Passengers')
    plt.xticks(rotation=0)
    plt.legend(title='Age Group')
    plt.tight_layout()
    plt.show()


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'

    train_df, test_df, merged_df = load_and_merge_data(train_path, test_path)

    print_data_count(train_df, test_df, merged_df)

    transported_df = preprocess_transported(merged_df)

    find_most_related_feature(transported_df)
    plot_transport_by_age_group(transported_df)
    plot_destination_age_distribution(merged_df)


if __name__ == '__main__':
    main()