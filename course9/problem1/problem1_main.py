from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# f1_score는 precision과 recall의 조화 평균을 계산하는 함수입니다.(역수를 더한 값)
from sklearn.metrics import confusion_matrix
# confusion_matrix는 TP, FP, FN, TN을 계산하는 함수입니다.
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import pandas as pd






def print_line():
    print('-' * 60)


def load_iris_dataset():
    iris_dataset = load_iris()

    return iris_dataset


def print_iris_description(iris_dataset):
    print_line()
    print('1. Iris 데이터셋 설명')
    print_line()
    print(iris_dataset.DESCR)


def print_iris_basic_info(iris_dataset):
    print_line()
    print('2. Iris 데이터셋 기본 정보')
    print_line()

    print('target_names')
    print(iris_dataset.target_names)

    print()
    print('feature_names')
    print(iris_dataset.feature_names)


def print_data_info(iris_dataset):
    data = iris_dataset.data

    print_line()
    print('3. data 항목 정보')
    print_line()

    print('데이터 모양:', data.shape)
    print('데이터 차원:', data.ndim)
    print('데이터 타입:', data.dtype)

    print()
    print('앞에서부터 5개의 데이터')
    print(data[:5])


def print_target_info(iris_dataset):
    target = iris_dataset.target

    print_line()
    print('4. target 항목 정보')
    print_line()

    print('데이터 모양:', target.shape)
    print('데이터 차원:', target.ndim)
    print('데이터 타입:', target.dtype)

    print()
    print('앞에서부터 5개의 데이터')
    print(target[:5])


def draw_iris_distribution_graph(iris_dataset):
    data = iris_dataset.data
    target = iris_dataset.target
    target_names = iris_dataset.target_names

    plt.figure(figsize=(8, 6))

    for target_index, target_name in enumerate(target_names):
        target_data = data[target == target_index]

        plt.scatter(
            target_data[:, 0],
            target_data[:, 1],
            label=target_name
        )

    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title('Iris Data Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()


def split_iris_data(iris_dataset):
    data = iris_dataset.data
    target = iris_dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size=0.25,
        random_state=0
    )

    return x_train, x_test, y_train, y_test


def print_train_test_shape(x_train, x_test, y_train, y_test):
    print_line()
    print('5. train_test_split 결과')
    print_line()

    print('X_train 모양:', x_train.shape)
    print('X_test 모양:', x_test.shape)
    print('y_train 모양:', y_train.shape)
    print('y_test 모양:', y_test.shape)

    print()
    print('X_train 크기:', x_train.size)
    print('X_test 크기:', x_test.size)
    print('y_train 크기:', y_train.size)
    print('y_test 크기:', y_test.size)


def train_knn_model(x_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(x_train, y_train)

    return knn_model


def predict_sample(knn_model, iris_dataset):
    sample_data = [[5, 2.9, 1, 0.2]]
    prediction = knn_model.predict(sample_data)
    predicted_name = iris_dataset.target_names[prediction[0]]

    print_line()
    print('6. KNeighborsClassifier 예측 결과')
    print_line()

    print('예측에 사용한 데이터:', sample_data)
    print('예측 결과 번호:', prediction[0])
    print('예측 결과 이름:', predicted_name)


def evaluate_model(knn_model, x_train, x_test, y_train, y_test):
    train_score = knn_model.score(x_train, y_train)
    test_score = knn_model.score(x_test, y_test)

    print_line()
    print('7. 학습된 모델 평가')
    print_line()

    print('train 데이터 정확도:', train_score)
    print('test 데이터 정확도:', test_score)


def main():
    iris_dataset = load_iris_dataset()

    print_iris_description(iris_dataset)
    print_iris_basic_info(iris_dataset)
    print_data_info(iris_dataset)
    print_target_info(iris_dataset)

    draw_iris_distribution_graph(iris_dataset)

    x_train, x_test, y_train, y_test = split_iris_data(iris_dataset)

    print_train_test_shape(x_train, x_test, y_train, y_test)

    knn_model = train_knn_model(x_train, y_train)

    predict_sample(knn_model, iris_dataset)

    evaluate_model(knn_model, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
