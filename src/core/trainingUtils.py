from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from core.config import config
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, r2_score, mean_absolute_error, recall_score
from sklearn.metrics._regression import root_mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np

import logging
import random

log = logging.getLogger(__name__)

'''
takes dataframe, time frame, and percent change and determines the number of buys/sells in the dataframe
saves these as 1 - buy and 0 wait in dataframe
'''
def get_buy_instances(df, timeframe, pct_change, tradingFrame):
    df = df.copy() 
    log.debug(f"getting buy signals for {timeframe}-{tradingFrame} with {pct_change*100:.2f}% change.")

    # calculate the percentage change over the timeframe
    buy_signals = []
    for i in range(len(df['close']) - timeframe):
        if tradingFrame == 'intraday':
            # if current percent change is greater than desired pct change and in the same day, mark as 1
            if (((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) >= pct_change) and (df.index[i].date() == df.index[i + timeframe].date()) and (df.index[i + timeframe].hour > 9):
                buy_signals.append(1)
            else:
                buy_signals.append(0)
        elif tradingFrame == 'daily':
            if (((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) >= pct_change):
                buy_signals.append(1)
            else:
                buy_signals.append(0)
        else:
            log.error("error: invalid trading frame - cannot get buy signals")
            return

    # fill remaining values with 0's 
    buy_signals.extend([0] * timeframe)

    if tradingFrame == 'intraday':
        log.debug(f"number of days: {len(df.index) / 11}")
    else:
        log.debug(f"number of days: {len(df.index)}")

    return buy_signals

# same as buy instances but checks if percent change is LESS THAN given pct change
def get_sell_instances(df, timeframe, pct_change, tradingFrame):
    df = df.copy() 
    log.debug(f"getting sell signals for {timeframe}-{tradingFrame} with {pct_change*100:.2f}% change.")

    # calculate the percentage change over the timeframe
    sell_signals = []
    for i in range(len(df['close']) - timeframe):
        if tradingFrame == 'intraday':
            # if current percent change is greater in magnitude than desired pct change and in the same day, mark as 1
            if (((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) <= pct_change) and (df.index[i].date() == df.index[i + timeframe].date()) and (df.index[i + timeframe].hour > 9):
                sell_signals.append(1)
            else:
                sell_signals.append(0)
        elif tradingFrame == 'daily':
            if (((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) <= pct_change):
                sell_signals.append(1)
            else:
                sell_signals.append(0)
        else:
            log.error("error: invalid trading frame - cannot get sell signals")
            return

    # fill remaining values with 0's 
    sell_signals.extend([0] * timeframe)

    if tradingFrame == 'intraday':
        log.debug(f"number of days: {len(df.index) / 11}")
    else:
        log.debug(f"number of days: {len(df.index)}")

    return sell_signals

'''
takes dataframe and buy signals
gets a random set of feature combinations
returns the most accurate and precise combination of features

@param df - the dataframe containing the buy signals and our features
@param buy_signal_col_name - name of the buy signal - needed so we can drop it when creating feature combinations
@param training_fxn - function used to generate model - either random train or sequential
@param balanced - bool - set class_weight in random forest model to balanced if specified
@param num_combs - the number of combinations we want to test
@params min_comb_len/max_comb_len - min/max number of features to use
'''
def get_best_feature_combination(df, signal_col_name, training_fxn, balanced, num_combs, min_comb_len, max_comb_len, num_threads=15):
    try:
        df_copy = df.copy()
        df_copy.drop(columns=[signal_col_name], inplace=True)  # remove y-var fields for feature combinations
        log.debug(f"features: {df_copy.columns.tolist()}")

        # get a random set of combinations of features and save as list of lists the save as chunks for threadpool
        feature_combinations = generate_combinations(lst=df_copy.columns.tolist(), num=num_combs, min=min_comb_len, max=max_comb_len)
        feature_combination_chunks = []
        chunk_size = len(feature_combinations) // num_threads
        for i in range(0, len(feature_combinations), chunk_size):
            feature_combination_chunks.append(feature_combinations[i : i + chunk_size])
        log.debug(f"thread chunk size: {chunk_size} | number of thread chunks: {len(feature_combination_chunks)} | listed number of entered combinations: {num_combs}")
        # use multithreading to split up work
        split_results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk in feature_combination_chunks:
                futures.append(executor.submit(evaluate_combinations, df, signal_col_name, training_fxn, balanced, chunk))

            for future in as_completed(futures):
                split_results.append(future.result())

        # get best results
        best_features = []
        best_prec = 0
        best_acc = 0
        for result in split_results:
            acc = result[0]
            precision = result[1]
            features = result[2]

            if (precision + acc > best_prec + best_acc and precision >= config.models.minPrecision and acc >= config.models.minAccuracy and precision <= config.models.maxPrecision):
                best_prec = precision
                best_acc = acc
                best_features = features

        log.debug(f"best precision: {best_prec}")
        log.debug(f"best accuracy: {best_acc}")
        log.debug(f"best features: {best_features}")

        return best_features, best_prec, best_acc
    except Exception as err:
        log.error(f"error getting best feature combination: {err}", exc_info=True)
    return None, None, None
'''
takes dataframe, buy signals, and list of feature combinations
trains random forests model on each feature set
returns the set of features that has best precision and accuracy

@param df - dataframe containing features and buy signals
@param buy_signal_col_name - y variable for our ai model
@param training_fxn - function used to generate model - either random train or sequential
@param balanced - bool - set class_weight in random forest model to balanced if specified
@param combs - a list of feature combinations to be tested
'''
def evaluate_combinations(df, signal_col_name, training_fxn, balanced, combs):
    try:
        best_acc = 0
        best_precision = 0
        best_features = []
        for c in combs:
            # NOTE: testing with timebased training
            overall_acc, acc, precision, features = training_fxn(df=df, features=c, target_y=signal_col_name, balanced=balanced, fileName=None)
            if precision is None: 
                continue

            if (precision + acc > best_precision + best_acc and precision >= config.models.minPrecision and acc >= config.models.minAccuracy and precision <= config.models.maxPrecision):
                best_precision = precision
                best_acc = acc
                best_features = features
        return best_acc, best_precision, best_features
    except Exception as err:
        log.error(f"error evaluating combinations: {err}", exc_info=True)
        return None, None, None
    
'''
trains random forests model
returns accuracy and precision of guessing buys

@param df - dataframe containing features and buy signals
@param features - the features to use in this model
@param target_y - the buy signals we are trying to predict
@param balanced - bool - set class_weight in random forest model to balanced if specified
@param fileName (optional) - if provided, save a file containing this model for future use
'''
def train_model(df, features, target_y, balanced, fileName):
    try:
        # Define your features and target variable
        X = df[features]
        y = df[target_y]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the Random Forest model
        if balanced:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else: 
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate per-class accuracy
        # cm[0,0] = True Negatives, cm[0,1] = False Positives
        # cm[1,0] = False Negatives, cm[1,1] = True Positives
        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Accuracy for class 0
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # Accuracy for class 1

        # Calculate precision specifically for predicting class 1
        precision_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0

        # Evaluate the model
        overall_accuracy = accuracy_score(y_test, y_pred)

        if (precision_1 >= config.models.minPrecision and class_1_accuracy >= config.models.minAccuracy and precision_1 <= config.models.maxPrecision):
            log.debug(f"Precision for predicting 1: {precision_1:.2f}")
            log.debug(f"associated features: {features}")
            log.debug(f"overall accuracy: {overall_accuracy}")
            log.debug(f"class 1 accuracy: {class_1_accuracy}")

        # save model if symbol provided
        if fileName:
            dump(rf_model, f'models/{fileName}.joblib')

        # Return metrics
        return overall_accuracy, class_1_accuracy, precision_1, features
    except Exception as err:
        log.error(f"error training and testing model: {err}", exc_info=True)
        return None, None, None, None
    
'''
    same as train_model but uses most recent data as testing 
    @param df 
    @param features
    @param target_y
    @param balanced - bool - set class_weight in random forest model to balanced if specified
    @param filename - string - if not null, save model
'''
def seq_train_model(df, features, target_y, balanced, fileName):
    try:
        X = df[features]
        y = df[target_y]
        
        # time-based sampling
        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # no y-vars are present
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            log.warning(f"no y-vars available for sequential sampling - cant train model")
            return None, None, None, None
        
        if balanced:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        else: 
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        precision_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
        
        overall_accuracy = accuracy_score(y_test, y_pred)
        
        if (precision_1 >= config.models.minPrecision and class_1_accuracy >= config.models.minAccuracy and precision_1 <= config.models.maxPrecision):
            log.debug(f"Precision for predicting 1: {precision_1:.2f}")
            log.debug(f"associated features: {features}")
            log.debug(f"overall accuracy: {overall_accuracy}")
            log.debug(f"class 1 accuracy: {class_1_accuracy}")
        
        if fileName:
            dump(rf_model, f'models/{fileName}.joblib')

        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'features': features,
            'importance': importances,
        }).sort_values(by='importance', ascending=False)
        log.debug(f"rf model feature importance: {feature_importance_df}")
        
        return overall_accuracy, class_1_accuracy, precision_1, features
    except Exception as err:
        log.error(f"error training and testing model: {err}", exc_info=True)
        return None, None, None, None

def seq_train_regr_model(df, features, y_var, fileName):
    try:
        X = df[features]
        y = df[y_var]

        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
        direction_acc = (np.sign(y_pred) == np.sign(y_test)).mean()

        if r2 > 0.15 and mae < 0.025 and rmse < 0.035 and direction_acc > 0.5:
            return r2, mae, rmse, direction_acc

        # importances = model.feature_importances_
        # plt.barh(features, importances)
        # plt.title("Feature importances")
        # plt.xlabel("importance")
        # plt.show()

        return None, None, None, None
    except Exception as err:
        log.error(f"error training rf regressor model: {err}", exc_info=True)
        return 
    
def seq_train_lin_regr_model(df, features, y_var, fileName):
    try:
        X = df[features]
        y = df[y_var]   

        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = LinearRegression().fit(X=X_train, y=y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        if r2 > 0.15 and mae < 0.025:
            return r2, mae, slope, intercept
        
        return None, None, None, None
    except Exception as err:
        log.error(f"error creating linear regression model: {err}", exc_info=True)

def seq_train_log_regr_model(df, features, y_var, fileName):
    try:
        X = df[features]
        y = df[y_var]   

        split_index = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # no y-vars are present
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None, None, None, None

        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X=X_train, y=y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        prec = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        coef = model.coef_[0][0]

        if prec > 0.5:
            return acc, prec, recall, coef
        
        return None, None, None, None
    except Exception as err:
        log.error(f"error creating linear regression model: {err}", exc_info=True)
'''
generates a list of unique entry combinations
@param lst - list of entries
@param num - number of combinations we want
@param min/max - min/max number of entries in a single combination

NOTE: if lst is not long enough, this will result in infinite loop
'''
def generate_combinations(lst, num, min, max):
    try:
        if max > len(lst):
            max = len(lst)

        combinations = set() # use set to ensure uniqueness
        i = 0 # safety measure to ensure we dont get stuck in infinite loop
        
        # iterate until we get num combinations or we've iterated num*2 times (to ensure we dont get stuck in loop forever for small lists)
        while len(combinations) < num and i < num*2:
            # randomly pick size between min and max
            size = random.randint(min, max)
            
            # get combination - need to to use tuple so its hashable
            comb = tuple(random.sample(lst, size))

            # add to combinations set
            combinations.add(comb)
            i += 1
        # return list of lists with combinations
        return [list(comb) for comb in combinations] # done this way to convert all tuples to lists
    except Exception as err:
        log.error(f"error generating combinations: {err}", exc_info=True)