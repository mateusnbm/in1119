#
#
#

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

cols_grades = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO"
]

cols_features = [
    "CO_UF_RESIDENCIA",
    "NU_IDADE",
    "TP_SEXO",
    "TP_ESTADO_CIVIL",
    "TP_COR_RACA",
    "CO_UF_NASCIMENTO",
    "TP_ST_CONCLUSAO",
    "TP_ANO_CONCLUIU",
    "TP_ENSINO",
    "IN_TREINEIRO",
    "CO_UF_ESC",
    "TP_LOCALIZACAO_ESC",
    "IN_BAIXA_VISAO",
    "IN_CEGUEIRA",
    "IN_SURDEZ",
    "IN_DEFICIENCIA_AUDITIVA",
    "IN_SURDO_CEGUEIRA",
    "IN_DEFICIENCIA_FISICA",
    "IN_DEFICIENCIA_MENTAL",
    "IN_DEFICIT_ATENCAO",
    "IN_DISLEXIA",
    "IN_DISCALCULIA",
    "IN_AUTISMO",
    "IN_VISAO_MONOCULAR",
    "IN_OUTRA_DEF",
    "CO_UF_PROVA",
    "CO_PROVA_CN",
    "CO_PROVA_CH",
    "CO_PROVA_LC",
    "CO_PROVA_MT",
    "Q001",
    "Q002",
    "Q003",
    "Q004",
    "Q005",
    "Q006",
    "Q007",
    "Q008",
    "Q009",
    "Q010",
    "Q011",
    "Q012",
    "Q013",
    "Q014",
    "Q015",
    "Q016",
    "Q017",
    "Q018",
    "Q019",
    "Q020",
    "Q021",
    "Q022",
    "Q023",
    "Q024",
    "Q025",
    "Q026",
    "Q027"
]

dataset = pd.read_csv("enem_2018.csv", delimiter=';', encoding="ISO-8859-1")
dataset['NU_NOTA_MEAN'] = dataset[cols_grades].sum(axis=1) * 0.2
dataset = dataset[(dataset.TP_PRESENCA_CN == 1) & (dataset.TP_PRESENCA_CH == 1) & (dataset.TP_PRESENCA_LC == 1) & (dataset.TP_PRESENCA_MT == 1) & (dataset.TP_PRESENCA_MT == 1)]
dataset = dataset.dropna()

dataset_data = dataset[cols_features]
dataset_target = dataset["NU_NOTA_MEAN"]

dataset_data = pd.concat([
    dataset_data.select_dtypes([], ['object']),
    dataset_data.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
    ], axis=1).reindex_axis(dataset_data.columns, axis=1)
cat_columns = dataset_data.select_dtypes(['category']).columns
dataset_data[cat_columns] = dataset_data[cat_columns].apply(lambda x: x.cat.codes)

features, targets = shuffle(dataset_data, dataset_target, random_state=10)

offset = int(features.shape[0] * 0.8)
features_train, targets_train = features[:offset], targets[:offset]
features_test, targets_test = features[offset:], targets[offset:]
params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'min_samples_split': 2,
    'learning_rate': 0.001,
    'loss': 'ls'
    }

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(features_train, targets_train)
mse = mean_squared_error(targets_test, clf.predict(features_test))

print("MSE: %.4f" % mse)

# ...

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(features_test)):
    test_score[i] = clf.loss_(targets_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(
    np.arange(params['n_estimators']) + 1,
    clf.train_score_, 'b-',
    label='Training Set Deviance'
    )
plt.plot(np.arange(
    params['n_estimators']) + 1,
    test_score, 'r-',
    label='Test Set Deviance'
    )
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# ...

names = np.array(cols_features)
feature_importance = clf.feature_importances_
names, feature_importance = zip(*sorted(zip(names, feature_importance)))
names = np.array(names[:10])
feature_importance = np.array(feature_importance[:10])

feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
