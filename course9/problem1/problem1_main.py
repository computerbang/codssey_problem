from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from sklearn.metrics import percision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# f1_score는 precision과 recall의 조화 평균을 계산하는 함수입니다.(역수를 더한 값)
from sklearn.metrics import confusion_matrix
# confusion_matrix는 TP, FP, FN, TN을 계산하는 함수입니다.
from sklearn.metrics import confusionMatrixDisplay
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
