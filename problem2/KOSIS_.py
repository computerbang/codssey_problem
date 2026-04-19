import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# 한글 폰트 설정 (경고 없이 실행하기 위해 필수)
matplotlib.rcParams['font.family'] = 'AppleGothic'  # MacOS에서는 'AppleGothic'을 사용, Windows에서는 'Malgun Gothic'을 사용
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    # 헤더를 설정하면 데이터가 4번째부터 있는거임.
    df = pd.read_csv(file_path, encoding='utf-8', header=[0, 1, 2, 3])
    return df


def filter_data(df):
    # 필요한 열만 선택
    result_ilban = []
    for col in df.columns:
        if '일반가구원' in col:
            result_ilban.append(col)

    result_sijeom = []
    for col in df.columns:
        if '시점' in col:
            result_sijeom.append(col)

    result_filter = result_sijeom + result_ilban
    df_filtered = df[result_filter]
    return df_filtered


def get_gender_stats(df):
    # 남자 합계 컬럼 찾기
    male_col = []
    for col in df.columns:
        if col[1] == '남자' and col[2] == '합계':
            male_col.append(col)

    # 여자 합계 컬럼 찾기
    female_col = []
    for col in df.columns:
        if col[1] == '여자' and col[2] == '합계':
            female_col.append(col)

    # 시점 컬럼 찾기
    sijeom_col = []
    for col in df.columns:
        if '시점' in col:
            sijeom_col.append(col)

    # 시점 + 남자 + 여자 컬럼만 추출
    target_cols = sijeom_col + male_col + female_col
    result = df[target_cols].copy()

    # 컬럼 이름 단순화
    result.columns = ['시점', '남자', '여자']

    # 시점을 인덱스로 설정
    result = result.set_index('시점')

    return result


def get_age_stats(df):
    # 제외할 연령 (전체 합계와 대분류는 빼고 세부 연령만 남긴다)
    exclude_ages = ['합계', '15~64세', '65세이상']

    # 성별이 '계'이고 연령이 세부 연령인 컬럼만 찾기
    age_cols = []
    for col in df.columns:
        if col[1] == '계' and col[2] not in exclude_ages and col[2] != '시점':
            age_cols.append(col)

    # 시점 컬럼 찾기
    sijeom_col = []
    for col in df.columns:
        if '시점' in col:
            sijeom_col.append(col)

    # 시점 + 연령별 컬럼만 추출
    target_cols = sijeom_col + age_cols
    result = df[target_cols].copy()

    # 컬럼 이름 단순화 (연령만 남기고 '시점'은 그대로)
    new_columns = ['시점']
    for col in age_cols:
        new_columns.append(col[2])
    result.columns = new_columns

    # 시점을 인덱스로 설정
    result = result.set_index('시점')

    return result


def get_gender_age_stats(df, gender):
    # 제외할 연령 (전체 합계와 대분류는 빼고 세부 연령만 남긴다)
    exclude_ages = ['합계', '15~64세', '65세이상']

    # 특정 성별 + 세부 연령별 컬럼 찾기
    age_cols = []
    for col in df.columns:
        if col[1] == gender and col[2] not in exclude_ages and col[2] != '시점':
            age_cols.append(col)

    # 시점 컬럼 찾기
    sijeom_col = []
    for col in df.columns:
        if '시점' in col:
            sijeom_col.append(col)

    # 시점 + 해당 성별 연령별 컬럼만 추출
    target_cols = sijeom_col + age_cols
    result = df[target_cols].copy()

    # 컬럼 이름 단순화
    new_columns = ['시점']
    for col in age_cols:
        new_columns.append(col[2])
    result.columns = new_columns

    # 시점을 인덱스로 설정
    result = result.set_index('시점')

    return result


def draw_gender_age_graph(male_df, female_df):
    # 남자와 여자 꺾은선 그래프를 좌우로 나란히 그리기
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 남자 그래프
    male_df.plot(ax=axes[0], marker='o')
    axes[0].set_title('남자 연령별 일반가구원 추이')
    axes[0].set_xlabel('연도')
    axes[0].set_ylabel('인구 (명)')
    axes[0].grid(True)
    axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=8, ncol=1)

    # 여자 그래프
    female_df.plot(ax=axes[1], marker='o')
    axes[1].set_title('여자 연령별 일반가구원 추이')
    axes[1].set_xlabel('연도')
    axes[1].set_ylabel('인구 (명)')
    axes[1].grid(True)
    axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=8, ncol=1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 상대경로 임, '../kosis_data.csv'
    # 상위 폴더에 있는 파일을 불러올 때는 '../'를 사용함.
    file_path = 'problem.csv'
    data = load_data(file_path)
    print('=== 원본 데이터 ===')
    print(data)

    print('\n=== 일반가구원 필터링 데이터 ===')
    filtered_data = filter_data(data)
    print(filtered_data)

    print('\n=== 남자/여자 연도별 일반가구원 통계 ===')
    gender_stats = get_gender_stats(filtered_data)
    print(gender_stats)

    print('\n=== 연령별 연도별 일반가구원 통계 ===')
    age_stats = get_age_stats(filtered_data)
    print(age_stats)

    # 꺾은선 그래프 (남자/여자 연령별)
    male_age_stats = get_gender_age_stats(filtered_data, '남자')
    female_age_stats = get_gender_age_stats(filtered_data, '여자')
    draw_gender_age_graph(male_age_stats, female_age_stats)