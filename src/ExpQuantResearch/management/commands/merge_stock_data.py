# merge_stock_data.py

from concurrent.futures import ThreadPoolExecutor
from core.config import config
from django.core.management.base import BaseCommand
from core.utils import plot_standardized_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from core.newsUtils import get_news_from_db
from core.intraDayUtils import get_intra_day_data_from_db
from core.socialUtils import get_social_from_db
from core.pressUtils import get_press_from_db
from core.insiderUtils import get_insider_from_db
from core.earningsUtils import get_earnings_from_db
from core.macroUtils import get_macro_from_db
from itertools import combinations
import pandas as pd
import MySQLdb
import logging
import random

log = logging.getLogger(__name__)


'''
gets data from db and merges into one db 
    - quicker than doing api calls


to run test:
$python3 src/manage.py merge_stock_data
'''
class Command(BaseCommand):
    help = 'Updates the stock data'

    def handle(self, *args, **options):
        try: 
            df_monthly_macro, df_monthly_macro_pct, df_monthly_macro_exp = get_macro_from_db(freq='monthly')
            df_quarterly_macro, df_quarterly_macro_pct, df_quarterly_macro_exp = get_macro_from_db(freq='quarterly')
            df_annual_macro, df_annual_macro_pct, df_annual_macro_exp = get_macro_from_db(freq='annual')

            # we want only 1 macro df
            df_macro = df_monthly_macro_exp.join(df_quarterly_macro_exp, how='inner').join(df_annual_macro_exp, how='inner')
            plot_standardized_data(df_macro, title='macro data')

            df_qearnings, df_pct_qearnings, df_exp_qearnings = get_earnings_from_db()
            df_insider, df_insider_pct, norm_df_insider, norm_df_insider_pct = get_insider_from_db()
            df_press, df_press_pct, norm_df_press, norm_df_press_pct = get_press_from_db()
            df_intraday, norm_df_intraday, norm_df_intraday_roc, norm_df_intraday_acc, norm_df_intraday_pct, norm_df_intraday_roc_pct, norm_df_intraday_acc_pct = get_intra_day_data_from_db()
            df_news, norm_df_news, norm_df_news_roc, norm_df_news_acc, norm_df_news_pct, norm_df_news_roc_pct, norm_df_news_acc_pct = get_news_from_db()
            df_social, norm_df_social, norm_df_social_roc, norm_df_social_acc, norm_df_social_pct, norm_df_social_roc_pct, norm_df_social_acc_pct = get_social_from_db()

            # rename duplicate columns
            df_news = df_news.add_suffix('_news')
            df_social = df_social.add_suffix('_social')
            df_press = df_press.add_suffix('_press')
            df_insider = df_insider.add_suffix('_insider')
            df_exp_qearnings = df_exp_qearnings.add_suffix('_qearnings_exp')
            df_macro = df_macro.add_suffix('_macro')

            # get features we want from each df
            df_intraday = df_intraday[['close', 'pre_market']] # only need closing values and marked premarket hours for now to get the buy signals

            # get buy signals from closing stock prices    
            df_intraday['pct1_H5_buys'] = get_buy_instances(df=df_intraday, timeframe=5, pct_change=0.01) # best results so far
            plot_standardized_data(df_intraday, title='intraday data')

            # shrink df to shorten training time
            # df_news = df_news[['sentiment_volume_news', 'AS_All_news', 'AS_Medium_news', 'APD_All_news', 'APD_Long_news', 'SVS_All_news']]
            # df_press = df_press[['sentiment_volume_press', 'AS_Long_press', 'AS_Medium_press', 'PPQ_All_press', 'PPQ_Long_press', 'PPQ_Medium_press', 'PPQ_Short_press', 'svs_all_press', 'svs_short_press', 'sentiment_score_press', 'count_press', 'AS_Short_press', 'svs_long_press', 'svs_medium_press']]
            # df_social = df_social[['volume_social', 'as_all_social', 'as_medium_social', 'as_short_social', 'av_all_social', 'av_short_social', 'av_medium_social', 'sentiment_social', 'sentiment_volume_social', 'as_long_social', 'svs_all_social', 'svs_long_social']]
            # df_insider = df_insider[['dollar_volume_change_insider', 'transaction_count_insider', 'transactionsInLastQuarter_insider', 'tpq_all_insider', 'tpq_short_insider', 'advs_all_insider', 'advs_short_insider']]
           
            # get best features from each df then merge into one df
            df_news = df_news.join(df_intraday, how='inner').drop(columns='pre_market') # dropping premarket because it will be joined at end and dont know another way to keep it
            df_social = df_social.join(df_intraday, how='inner').drop(columns='pre_market')
            df_press = df_press.join(df_intraday, how='inner').drop(columns='pre_market')
            df_insider = df_insider.join(df_intraday, how='inner').drop(columns='pre_market')
            df_exp_qearnings = df_exp_qearnings.join(df_intraday, how='inner').drop(columns='pre_market')
            df_macro = df_macro.join(df_intraday, how='inner').drop(columns='pre_market')

            macro_features, macro_precision, macro_acc = get_best_feature_combination(df_macro, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
            qearnings_exp_features, qearnings_exp_precision, qearnings_exp_acc = get_best_feature_combination(df_exp_qearnings, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
            news_features, news_precision, news_acc = get_best_feature_combination(df_news, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
            social_features, social_precision, social_acc = get_best_feature_combination(df_social, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
            press_features, press_precision, press_acc = get_best_feature_combination(df_press, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
            insider_features, insider_precision, insider_acc = get_best_feature_combination(df_insider, 'pct1_H5_buys', num_combs=200, min_comb_len=5, max_comb_len=6)
        
            log.debug("best features")
            log.debug(f"features: {news_features} | precision: {news_precision} | acc: {news_acc}")
            log.debug(f"features: {qearnings_exp_features} | precision: {qearnings_exp_precision} | acc: {qearnings_exp_acc}")
            log.debug(f"features: {social_features} | precision: {social_precision} | acc: {social_acc}")
            log.debug(f"features: {press_features} | precision: {press_precision} | acc: {press_acc}")
            log.debug(f"features: {insider_features} | precision: {insider_precision} | acc: {insider_acc}")
            log.debug(f"features: {macro_features} | precision: {macro_precision} | acc: {macro_acc}")

            log.debug("getting best combination")

            # take best features and combine to find best combination of features
            df_news = df_news[news_features].copy()
            df_social = df_social[social_features].copy()
            df_press = df_press[press_features].copy()

            # get best features from short term data
            df_short_term = df_intraday.join(df_social, how='inner').join(df_press, how='inner').join(df_news, how='inner').drop(columns='pre_market')
            short_features, short_precision, short_acc = get_best_feature_combination(df_short_term, 'pct1_H5_buys', num_combs=200, min_comb_len=10, max_comb_len=15)
            
            df_insider = df_insider[insider_features].copy()
            df_exp_qearnings = df_exp_qearnings[qearnings_exp_features].copy()
            df_macro = df_macro[macro_features].copy()

            # get best features from long term data
            df_long_term = df_intraday.join(df_insider, how='inner').join(df_exp_qearnings, how='inner').join(df_macro, how='inner').drop(columns='pre_market')
            long_features, long_precision, long_acc = get_best_feature_combination(df_long_term, 'pct1_H5_buys', num_combs=200, min_comb_len=10, max_comb_len=15)

            # get best features from short and long term feature sets
            df_final = df_news.join(df_social, how='inner').join(df_press, how='inner').join(df_insider, how='inner').join(df_exp_qearnings, how='inner').join(df_macro, how='inner')
            combined_features = short_features + long_features
            df_final = df_final[combined_features].copy()
            df_final = df_intraday.join(df_final, how='inner')

            final_features, final_precision, final_acc = get_best_feature_combination(df_final, 'pct1_H5_buys', num_combs=500, min_comb_len=20, max_comb_len=25)
            log.debug("best features")
            log.debug(f"features: {final_features} | precision: {final_precision} | acc: {final_acc}")

        except Exception as err:
            log.warn(f"error retrieving and saving stock data: {err}", exc_info=True)


'''
takes dataframe, time frame, and percent change and determines the number of buys/sells in the dataframe
saves these as 1 - buy and 0 wait in dataframe
'''
def get_buy_instances(df, timeframe, pct_change):
    df = df.copy() 
    log.debug(f"getting buy signals for {timeframe}H with {pct_change*100:.2f}% change.")

    # calculate the percentage change over the timeframe
    buy_signals = []
    for i in range(len(df['close']) - timeframe):
        # if current percent change is greater than desired pct change and in the same day, mark as 1
        if (((df['close'][i + timeframe] - df['close'][i]) / df['close'][i]) >= pct_change) and (df.index[i].date() == df.index[i + timeframe].date()) and (df.index[i + timeframe].hour > 9):
            buy_signals.append(1)
        else:
            buy_signals.append(0)

    # fill remaining values with 0's 
    buy_signals.extend([0] * timeframe)

    log.debug(f"number of days: {len(df) / 11}")

    # get number of possible trades
    num_trades = 0
    for i in range(len(buy_signals) - 1):
        # series of 1's is a single trade window, so count that as 1 trade - if next value is 0, then we have a new trade window
        if (buy_signals[i] == 1 and buy_signals[i + 1] == 0):
            num_trades += 1

    log.debug(f"number of buys for {timeframe}-hour trades at {pct_change*100}% increase: {num_trades}")

    return buy_signals


'''
takes dataframe and buy signals
gets every combination of features in a list
creates multiple threads to do the following:
    - train and tests model on features and gets accuracy and precision
    - returns feature list with best precision
'''
def get_best_feature_combination(df, buy_signal_col_name, num_combs, min_comb_len, max_comb_len):
    try:
        df_copy = df.copy()
        df_copy.drop(columns=['close', buy_signal_col_name], inplace=True)  # remove y-var fields for feature combinations
        log.debug("features")
        log.debug(df_copy.columns.tolist())
        # get every combination of features and save as list of lists
        combs = generate_combinations(lst=df_copy.columns.tolist(), min=min_comb_len, max=max_comb_len)
        # it is impossible to test every combination of features so pick a random set of 'num_combs' as the list of lists of features to test
        random_combs = random.sample(combs, num_combs)

        # Split combinations into 4 chunks
        split_points = [len(random_combs) // 4 * i for i in range(5)]
        chunks = [random_combs[split_points[i]:split_points[i+1]] for i in range(4)]

        # Use ThreadPoolExecutor to process each chunk in parallel
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(evaluate_combinations, df, buy_signal_col_name, chunk) for chunk in chunks]
            results = [future.result() for future in futures]

        # Find the best overall result
        best_acc = 0
        best_precision = 0
        best_features = []
        for acc, precision, features in results:
            if (precision + acc > best_precision + best_acc):
                best_precision = precision
                best_acc = acc
                best_features = features

        log.debug(f"best precision: {best_precision}")
        log.debug(f"best accuracy: {best_acc}")
        log.debug(f"best features: {best_features}")

        return best_features, best_precision, best_acc
    except Exception as err:
        log.error(f"error getting best feature combination: {err}", exc_info=True)
    return
'''
takes dataframe, buy signals, and list of features 
trains random forests model on each feature set
returns the set of features that has best precision
'''
def evaluate_combinations(df, buy_signal_col_name, combs):
    try:
        best_acc = 0
        best_precision = 0
        best_features = []
        for c in combs:
            overall_acc, acc, precision, features = train_model(df=df, features=c, target_y=buy_signal_col_name)
            if (precision + acc > best_precision + best_acc):
                best_precision = precision
                best_acc = acc
                best_features = features
        return best_acc, best_precision, best_features
    except Exception as err:
        log.error(f"error evaluating combinations: {err}", exc_info=True)
        return
    
'''
trains random forests model
returns accuracy and precision of guessing buys
'''
def train_model(df, features, target_y):
    try:
        # Define your features and target variable
        # log.debug(f"training model for: {target_y} and {features}")

        X = df[features]
        y = df[target_y]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # log.debug("Confusion Matrix:")
        # log.debug(cm)

        # Calculate per-class accuracy
        # cm[0,0] = True Negatives, cm[0,1] = False Positives
        # cm[1,0] = False Negatives, cm[1,1] = True Positives
        class_0_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Accuracy for class 0
        class_1_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # Accuracy for class 1

        # log.debug(f"Accuracy for predicting 0: {class_0_accuracy:.2f}")
        # log.debug(f"Accuracy for predicting 1: {class_1_accuracy:.2f}")

        # Calculate precision specifically for predicting class 1
        precision_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0

        # Evaluate the model
        overall_accuracy = accuracy_score(y_test, y_pred)
        # log.debug(f"Overall Accuracy: {overall_accuracy:.2f}")
        # log.debug("Classification Report:")
        # log.debug(classification_report(y_test, y_pred))

        # Analyze feature importance
        # feature_importances = rf_model.feature_importances_
        # for feature, importance in zip(features, feature_importances):
        #     log.debug(f"{feature}: {importance:.4f}")

        if (precision_1 >= 0.8 and class_1_accuracy >= 0.5):
            log.debug(f"Precision for predicting 1: {precision_1:.2f}")
            log.debug(f"associated features: {features}")
            log.debug(f"overall accuracy: {overall_accuracy}")
            log.debug(f"class 1 accuracy: {class_1_accuracy}")
        # Return metrics
        return overall_accuracy, class_1_accuracy, precision_1, features
    except Exception as err:
        log.error(f"error training and testing model: {err}", exc_info=True)
        return None, None, None, None


def generate_combinations(lst, min, max):
    result = []
    # Loop through all possible lengths for combinations
    for r in range(min, max + 1):
        # Generate combinations of length `r`
        result.extend(combinations(lst, r))
    return [list(comb) for comb in result]