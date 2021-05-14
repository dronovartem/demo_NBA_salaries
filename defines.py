import pandas as pd
from sklearn.neighbors import NearestNeighbors

# dataset
DATASET_PATH = '2017-18_NBA_salary.csv'
SEASON_STATS_DATASET_PATH = 'nba_17_18.csv'
SALARY_PREDICTION_MODEL_PATH = 'salary_model.pkl'
KNN_MODEL_PATH = 'knn_model.pkl'

# base options able to set
DEFAULT_CONTROL_FEATURES = ["NBA_DraftNumber", "Age", "MP", "PER", "USG%", "BPM"]
DEFAULT_PICKED_ADVANCED_FEATURES = ["Age", "MP"]
INTEGER_FEATURES = ["NBA_DraftNumber", "Age"]
SEASON_GAMES_COUNT = 82 # i scale MP written as a season playing time to average per game

# set a name for default unfamous player
ABSTRACT_PLAYER = 'Абстрактный игрок'

# define bounds and initial filling for sliders
DEFAULT_FEATURES_RANGE = pd.DataFrame({
    'min': [  1 ,  18 ,   0. , -50.,   0. , -60.],
    'max': [  62 ,  45 , 48. , 150.,   60. ,  60.],
}, index=DEFAULT_CONTROL_FEATURES)

# define minimum salary for model
MODEL_MIN_VALUE = 46080

__FEATURES_EXPLANATION = [
    "Позиция игрока на драфте НБА перед его первым сезоном в лиге. ",
    "Возраст игрока на 1 февраля текущего сезона.",
    "Среднее количество минут за игру в предыдущем сезоне.",
    "Коэффициент эффективности игрока (PER) - PER суммирует все положительные достижения игрока, \
    вычитает отрицательные достижения и возвращает поминутный рейтинг эффективности игрока.",
    "Процент использования - это оценка процента командных игр, когда игрок находился на площадке.",
    "Плюс-минус на площадке."
]

FEATURES_DOC_STRING = {k: v for k, v in zip(DEFAULT_CONTROL_FEATURES, __FEATURES_EXPLANATION)}


# Extended class to work with kNN pipeline
class KNN(NearestNeighbors):
    def predict(self, X):
        return super().kneighbors(X,  return_distance=False)