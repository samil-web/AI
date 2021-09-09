import os
import time
from PIL import Image
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
from tqdm import tqdm

def img_to_csv():
    parent_dir = "Data"
    data = pd.DataFrame()
    for folder in os.listdir(parent_dir):

        print(folder)
        
        if folder == "train":
            for f in os.listdir(parent_dir+"/"+folder):

                class_data = np.zeros(  ( len(os.listdir(parent_dir+"/"+folder+"/"+f) ), 785) )
                print(class_data.shape)

                for i, img_name in enumerate(os.listdir(parent_dir+"/"+folder+"/"+f)):
                    
                    img = Image.open(parent_dir+"/"+folder+"/"+f+"/"+img_name)
                    img_arr = np.array(img, dtype=int)
                    img_arr = img_arr.flatten()
                    class_data[i,:784] = img_arr
                    class_data[i,784] = int(f)

                class_data = pd.DataFrame(class_data)
                data = pd.concat([data, class_data])
    data.to_csv("train.csv", index=False)
    return data


data = img_to_csv()


x = data.values[:,:784]
y = data.values[:,784]
print(x.shape)
print(y.shape)

tree_classifiers = {
  #"Decision Tree": DecisionTreeClassifier(),
  #"Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  #"AdaBoost":      AdaBoostClassifier(n_estimators=100),
  #"Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  #"Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
  #"XGBoost":       XGBClassifier(n_estimators=100),
  #"LightGBM":      LGBMClassifier(n_estimators=100),
  #"CatBoost":      CatBoostClassifier(n_estimators=100),
}



x_train, x_val, y_train, y_val = model_selection.train_test_split(
    x, y,
    test_size=0.2,
    stratify = y,   # ALWAYS RECOMMENDED FOR BETTER VALIDATION
    random_state=4  # Recommended for reproducibility
)


results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})


for model_name, model in tqdm(tree_classifiers.items()):
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
### END SOLUTION


results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
print(results_ord)

