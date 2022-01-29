import ast
import os 
import sys
import json
import yaml
import warnings
from joblib import dump
from loguru import logger
from shutil import copy

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from helpers import load_data, process_data, random_train_test_split

warnings.filterwarnings('ignore')
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
logger.add("../logs/{time}.log")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")

    
if __name__ == "__main__":
    data, target = load_data(data_folder="../dataset/")
    
    with open('../dataset/stopwords.yaml', 'r') as f:
        curated_stop_words = yaml.safe_load(f)

    processed_data = process_data(data, stemming=True, stopwords_locale='german', curated_stop_words=curated_stop_words)
    
    train_x, test_x, train_y, test_y = random_train_test_split(processed_data, target)
    
    metrics_list = []
    
    # LogisticRegression
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("logreg", LogisticRegression(n_jobs=-1))])
    with open("../gridsearch_space/model_logreg.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['LogisticRegression param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)
    
    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_logreg', precision, recall, fscore])

    with open('../configs/logreg.json', 'w') as file:
        file.write(json.dumps(best_params))
    
    logger.info(['LogisticRegression', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe, filename="../trained_models/model_logreg.sav")

    # DecisionTreeClassifier
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("dtree", DecisionTreeClassifier())])

    with open("../gridsearch_space/model_dectree.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['LogisticRegression param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)
    
    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_dectree', precision, recall, fscore])
    
    with open('../configs/dectree.json', 'w') as file:
        file.write(json.dumps(best_params))

    logger.info(['DecisionTreeClassifier', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe, filename="../trained_models/model_dectree.sav")
    
    # KNeighborsClassifier
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("knc", KNeighborsClassifier(n_jobs=-1))])
    with open("../gridsearch_space/model_kneg.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['KNeighborsClassifier param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)
    
    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_kneg', precision, recall, fscore])

    with open('../configs/kneg.json', 'w') as file:
         file.write(json.dumps(best_params))

    logger.info(['KNeighborsClassifier', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe, filename="../trained_models/model_kneg.sav")

    # SGDClassifier
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("sgd", SGDClassifier(n_jobs=-1))])
    with open("../gridsearch_space/model_sgd.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['SGDClassifier param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)

    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_sgd', precision, recall, fscore])

    with open('../configs/sgd.json', 'w') as file:
         file.write(json.dumps(best_params))
            
    logger.info(['SGDClassifier', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe, filename="../trained_models/model_sgd.sav")
    
    # GradientBoostingClassifier
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("gbc", GradientBoostingClassifier())])
    with open("../gridsearch_space/model_gradboost.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['GradientBoostingClassifier param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)
    
    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_gradboost', precision, recall, fscore])

    with open('../configs/gradboost.json', 'w') as file:
         file.write(json.dumps(best_params))
    
    logger.info(['GradientBoostingClassifier', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe_clf, filename="../trained_models/model_gradboost.sav")
    
    # RandomForestClassifier
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("rfc", RandomForestClassifier(n_jobs=-1))])
    with open("../gridsearch_space/model_ranfor.json", "r") as gs_file:
        contents = gs_file.read()
        params = ast.literal_eval(contents)
        logger.info(['RandomForestClassifier param space', params])

    pipe_clf = GridSearchCV(pipe, params, n_jobs=-1, scoring="f1_macro", verbose=1)
    pipe_clf.fit(processed_data, target)
    best_params = pipe_clf.best_params_
    pipe.set_params(**best_params).fit(train_x, train_y)
    pipe_pred = pipe.predict(test_x)
    report = classification_report(test_y, pipe_pred)

    precision, recall, fscore, support = precision_recall_fscore_support(test_y, pipe_pred, average='macro')
    metrics_list.append(['model_ranfor', precision, recall, fscore])

    with open('../configs/ranfor.json', 'w') as file:
         file.write(json.dumps(best_params))

    logger.info(['RandomForestClassifier', best_params, report])
    pipe.set_params(**best_params).fit(processed_data, target)
    dump(pipe, filename="../trained_models/model_ranfor.sav")
    
    # Best trained model
    compare_model_metric = 'fscore'
    metrics_df = pd.DataFrame(metrics_list, columns=['model', 'precision', 'recall', 'fscore'])    
    metrics_df = metrics_df.sort_values(compare_model_metric, ascending=False)

    best_trained_model_filename = '../trained_models/' + metrics_df.loc[0]['model'] + '.sav'

    copy(best_trained_model_filename, '../trained_models/best_model.sav')
    logger.info(f'Copied {best_trained_model_filename} as best_model.sav')
