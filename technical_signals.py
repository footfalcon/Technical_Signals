# --- Evaluating Technical Signals -------------------------------------------
''' Get top/bottom outlier stats on all columns (features), eg: features values
    2-sd, get dates, count how many times that feature has reached/exceeded
    such levels, what were the subsequent 1 week/month/qtr returns. '''

import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd


# -- Set up data -------------------------------------------------------------
# Stock data
def get_data(ticker):
    fpath = '/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/stocks_features_standard/'
    fh = os.path.join(fpath, ticker+'.pkl')
    df = pd.read_pickle(fh)
    features = ['CLOSE', 'BBAND_PCT_STD', 'RSI_14', 'RSI_28', 'RSI_42', 'RSI_56',
                'MA_5_vs_20','MA_20_vs_50', 'MA_50_vs_100', 'PE1_STD', 'PB1_STD']
    df = df[features]

    return df


def get_universe():
    fh='/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/universe.pkl'
    df_universe = pd.read_pickle(fh)

    features = ['CLOSE', 'BBAND_PCT_STD', 'RSI_14', 'RSI_28', 'RSI_42', 'RSI_56',
                'MA_5_vs_20','MA_20_vs_50', 'MA_50_vs_100', 'PE1_STD', 'PB1_STD']
    df_universe = df_universe[features]

    return df_universe



# -- Wrap into functions and loop through all features -----------------------
# Global variables
stdev_threshold = 1.5
window = 2
#df_universe = get_universe()


def get_signal(df, feature, stdev_threshold, window):
    ''' Standardize feature and generate signal '''
    feat_stdize =  (df[feature] - df[feature].mean()) / df[feature].std()
    feat_signal = (feat_stdize > stdev_threshold).astype(int)
    feat_signal = feat_signal - (feat_stdize < -stdev_threshold).astype(int)

    # Get clean signal (minimize duplication of signals)
    shift = feat_signal.diff().shift(window)
    diff = feat_signal.diff(window).shift(1)  # should this always be shift(1)?
    buys = ((feat_signal + diff + shift) == 3).astype(int)  # should this always == 3?
    sells = -((feat_signal + diff + shift) == -3).astype(int)  # should this always == -3?
    clean_signal = buys + sells

    return clean_signal



def get_stats(df, feature, stdev_threshold, window):
    ''' Calculate week/month/quarter outcome returns, hit rate, accuracy etc. '''
    # get index for calculating returns
    clean_signal = get_signal(df, feature, stdev_threshold, window).reset_index(drop=True)
    close = df.CLOSE.reset_index(drop=True)
    index = ['Week', 'Month', 'Quarter']  #level_1 index
    cols = ['Freq', 'Hit Rate', 'Accuracy', 'Mean_Rtn', 'Exp_Rtn',
            'Signal_Score', 'Max', 'Min', 'Stdev']

    # Buys
    idx_buys = np.where(clean_signal == -1)[0]  # buy when feature is oversold == -1
    count = 0
    buy_stats = pd.DataFrame(0, index=index, columns=cols)  # Make df of zeros
    for days in [5, 21, 63]:
        l = []
        rtn = ( (close.shift(-days) * clean_signal) / (close * clean_signal) * 100 - 100).dropna()

        if idx_buys.size == 0:
            print('\t\tNo buys\t\t\tDays:', days)
            continue
        elif idx_buys[-1]+days < len(df):
            print('\t\tAll buys in range\tDays:', days)
            buy_rtn = [rtn[i] for i in idx_buys]
            freq = len(idx_buys)
            hit = np.sum(np.array(buy_rtn) > 0)
            accuracy = hit / freq
            hi = max(buy_rtn)
            low = min(buy_rtn)
            avg_buy_rtn = np.mean(buy_rtn)
            exp_buy_rtn = accuracy * avg_buy_rtn
            signal_score = freq * exp_buy_rtn
            stdev = np.std(buy_rtn)
            l.append([freq, hit, accuracy, avg_buy_rtn, exp_buy_rtn, signal_score, hi, low, stdev])
            buy_stats.loc[index[count]] = l[0]
        elif len(idx_buys[ : np.where(idx_buys+days >= len(df))[0][0] ]) == 0:
            print('\t\tBuys out of range\tDays:', days)
            break
        else:
            idx_buys = idx_buys[ : np.where(idx_buys+days >= len(df))[0][0] ]
            print('\t\t', len(idx_buys), ' buys in range\tDays:', days)
            buy_rtn = [rtn[i] for i in idx_buys]
            freq = len(idx_buys)
            hit = np.sum(np.array(buy_rtn) > 0)
            accuracy = hit / freq
            hi = max(buy_rtn)
            low = min(buy_rtn)
            avg_buy_rtn = np.mean(buy_rtn)
            exp_buy_rtn = accuracy * avg_buy_rtn
            signal_score = freq * exp_buy_rtn
            stdev = np.std(buy_rtn)
            l.append([freq, hit, accuracy, avg_buy_rtn, exp_buy_rtn, signal_score, hi, low, stdev])
            buy_stats.loc[index[count]] = l[0]
        count += 1
    buy_stats = buy_stats.stack()  # Make into multi-index
    buy_stats = buy_stats.to_frame(feature+'_buy')  # Convert from series to df and name col

    # Sells
    idx_sells = np.where(clean_signal == 1)[0]  # sell when feature is overbought == 1
    count = 0
    sell_stats = pd.DataFrame(0, index=index, columns=cols)  # Make df of zeros
    for days in [5, 21, 63]:
        l = []
        rtn = ( (close.shift(-days) * clean_signal) / (close * clean_signal) * 100 - 100).dropna()

        if idx_sells.size == 0:
            print('\t\tNo sells\t\t\tDays:', days)
            continue
        elif idx_sells[-1]+days < len(df):
            print('\t\tAll sells in range\tDays:', days)
            sell_rtn = [-rtn[i] for i in idx_sells]
            freq = len(idx_sells)
            hit = np.sum(np.array(sell_rtn) > 0)
            accuracy = hit / freq
            hi = max(sell_rtn)
            low = min(sell_rtn)
            avg_sell_rtn = np.mean(sell_rtn)
            exp_sell_rtn = accuracy * avg_sell_rtn
            signal_score = freq * exp_sell_rtn
            stdev = np.std(sell_rtn)
            l.append([freq, hit, accuracy, avg_sell_rtn, exp_sell_rtn, signal_score, hi, low, stdev])
            sell_stats.loc[index[count]] = l[0]
        elif len(idx_sells[ : np.where(idx_sells+days >= len(df))[0][0] ]) == 0:
            print('\t\Sells out of range\tDays:', days)
            break
        else:
            idx_sells = idx_sells[ : np.where(idx_sells+days >= len(df))[0][0] ]
            print('\t\t', len(idx_sells), ' sells in range\tDays:', days)
            sell_rtn = [rtn[i] for i in idx_sells]
            freq = len(idx_sells)
            hit = np.sum(np.array(sell_rtn) > 0)
            accuracy = hit / freq
            hi = max(sell_rtn)
            low = min(sell_rtn)
            avg_sell_rtn = np.mean(sell_rtn)
            exp_sell_rtn = accuracy * avg_sell_rtn
            signal_score = freq * exp_sell_rtn
            stdev = np.std(sell_rtn)
            l.append([freq, hit, accuracy, avg_sell_rtn, exp_sell_rtn, signal_score, hi, low, stdev])
            sell_stats.loc[index[count]] = l[0]
        count += 1
    sell_stats = sell_stats.stack()  # Make into multi-index
    sell_stats = sell_stats.to_frame(feature+'_sell')  # Convert from series to df and name col

    return buy_stats, sell_stats



def combine_stats(ticker, stdev_threshold, window):
    ''' Concat buy/sell signal df's and name cols (works but may not be most concise/efficient) '''
    df = get_data(ticker)
    features = list(df.columns)
    df_stats = pd.DataFrame()
    for f in features[1:]:
        print('\t', f)
        buy_stats, sell_stats = get_stats(df, f, stdev_threshold, window)
        df_stats = pd.concat([df_stats, buy_stats, sell_stats], axis=1)

    return df_stats



def get_signal_score(ticker):
    ''' Distill df_stats to a single score for each buy/sell indicator to be used
        as a weight in final dashboard ranking. Done by taking product of signal
        frequency, accuracy and mean_rtn (which is exp_rtn) '''
    df_stats = combine_stats(ticker, stdev_threshold, window)
    signal_score = df_stats.loc[(slice(None), 'Signal_Score'), :].reset_index(level=1, drop=True)
    signal_score.loc['Average'] = signal_score.mean()

    return signal_score
#signal_score = get_signal_score(df_stats)
#signal_score = signal_score.sort_values(by='Average', axis=1, ascending=False)


# Get signal_score_universe()
def get_signal_score_universe():
    fpath = '/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/stocks_features_standard/'
    tickers = sorted( [f[:-4] for f in listdir(fpath) if isfile(join(fpath, f))] )[1:]

    signal_score_universe = pd.DataFrame()
    for ticker in tickers:
        print(ticker)
        signal_score = get_signal_score(ticker)
        signal_score = pd.concat([signal_score], keys=[ticker], names=['Ticker'])
        signal_score_universe = pd.concat([signal_score_universe, signal_score], axis=0)

    return signal_score_universe
#signal_score_universe = get_signal_score_universe()



# --- Dashboard/ref part -----------------------------------------------------

def get_current_levels():
    #df = get_data('BBCA_IJ')  # any ticker in the universe will do; its just to get the feature set
    df_universe = get_universe()
    features = df_universe.columns[1:]
    stocks = df_universe.index.levels[1]

    current_df = pd.DataFrame(columns=features)
    for stock in stocks:
        current = df_universe.loc[(df_universe.index.levels[0][-1], stock), features]
        current = pd.DataFrame(current).T.reset_index(level=0, drop=True)
        current_df = current_df.append(current)

    return current_df
#current_df = get_current_levels()



def get_ranked_outliers(threshold):
    # Current stdev levels of features and their signal scores
    current_df = get_current_levels()
    signal_score_universe = get_signal_score_universe()
    index = current_df.index
    cols = current_df.columns

    # Setup dictionaries to hold results
    current_buy = current_df.copy()
    current_buy[current_buy > -threshold] = 0  # leave only outliers below -1.5 stdev threshold
    buy_dict = {}
    current_sell = current_df.copy()
    current_sell[current_sell < threshold] = 0  # leave only outliers above 1.5 stdev threshold
    sell_dict = {}

    for prd in ['Week', 'Month', 'Quarter', 'Average']:
        # Ranked buys
        buy_score = signal_score_universe.loc[ (slice(None), prd), : ].reset_index(level=1, drop=True)
        buy_score = buy_score.iloc[:, ::2]
        buy_score = buy_score.reindex(sorted(buy_score.columns), axis=1)
        buy_rank = pd.DataFrame(buy_score.values * current_buy.values, index=index, columns=cols)
        buy_rank['Sum'] = buy_rank.sum(axis=1)
        buy_rank = buy_rank.sort_values(by='Sum')
        buy_dict[prd] = buy_rank

        # Ranked sells
        sell_score = signal_score_universe.loc[ (slice(None), prd), : ].reset_index(level=1, drop=True)
        sell_score = sell_score.iloc[:, ::2]
        sell_score = sell_score.reindex(sorted(sell_score.columns), axis=1)
        sell_rank = pd.DataFrame(sell_score.values * current_sell.values, index=index, columns=cols)
        sell_rank['Sum'] = sell_rank.sum(axis=1)
        sell_rank = sell_rank.sort_values(by='Sum', ascending=False)
        sell_dict[prd] = sell_rank

    return buy_dict, sell_dict
buy_dict, sell_dict = get_ranked_outliers(threshold=1.5)



# --- Accuracy/return plots --------------------------------------------------
# Setup each Rtn_Pd df
def get_rtn_prd_df(ticker):
    rtn_prd = ['Week', 'Month', 'Quarter']
    rtn_prd_dict = {}

    df_stats = combine_stats(ticker, stdev_threshold, window)
    df_summary = df_stats.loc[(slice(None), ['Accuracy', 'Mean_Rtn', 'Exp_Rtn']), :]
    flat_df = df_summary.reset_index(level=1, drop=False)
    flat_df = flat_df.rename(columns={'level_1':'Stats'})
    flat_df.index.name = 'Rtn_Pd'
    flat_df_T = flat_df.T

    for prd in rtn_prd:
        df_prd = flat_df_T.loc[:, prd]
        df_prd.columns = ['Accuracy', 'Mean_Rtn', 'Exp_Rtn']
        df_prd.name = prd
        df_prd = df_prd[1:]
        df_prd['Accuracy'] = df_prd.Accuracy - 0.5
        df_prd = df_prd.sort_values(by='Accuracy', ascending=False)
        rtn_prd_dict[prd+'_df'] = df_prd

    return rtn_prd_dict


def plot_stats(ticker):
    rtn_prd_dict = get_rtn_prd_df(ticker)

    plt.style.use('ggplot')
    plt.figure(figsize=(18, 6), tight_layout=True)
    plt.rc('font', size=6)

    for i in range(len(rtn_prd_dict)):
        name = list(rtn_prd_dict.keys())[i]
        df = rtn_prd_dict[name]

        plt.subplot(1, 3, i+1)
        # first axis
        ax = df.Accuracy.plot.bar(color='gray')
        ax.set_title('1-'+name[:-3]+' Return Outcomes', weight='bold', fontsize=8)
        ax.set_ylabel('Accuracy: ppts above/below 50%', color='gray')
        ax.tick_params(axis='y', labelcolor='gray')
        ylow, yhigh = ax.get_ylim()
        # second axis
        ax2 = ax.twinx()  # instantiate second y-axis that shares the same x-axis
        ax2.plot(df.Exp_Rtn, 'b^', markersize=3)
        ax2.plot(df.Mean_Rtn, 'gs', markersize=3)
        _, ax2yhigh = ax2.get_ylim()  # these 2 lines set second y-axis range to match primary range...
        ax2.set_ylim(ax2yhigh *  ylow / yhigh , ax2yhigh)
        ax2.set_ylabel('Returns', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        plt.legend()
        plt.show()

    return None
#plot_stats('BBCA_IJ')


def plot_features(ticker):
    df = get_data(ticker)
    features = df.columns[1:]

    counter = 1
    fig = plt.figure(figsize=(18, 16), tight_layout=True)
    for feature in features:
        clean_signal = get_signal(df, feature, stdev_threshold, window)
        idx_buys = np.where(clean_signal == -1)[0]  # buy when feature is oversold == -1
        buys = [ df[feature][i] for i in idx_buys ]
        buy_dates = [ df.index[i] for i in idx_buys ]
        idx_sells = np.where(clean_signal == 1)[0]  # sell when feature is overbought == 1
        sells = [ df[feature][i] for i in idx_sells ]
        sell_dates = [ df.index[i] for i in idx_sells ]

        ax = fig.add_subplot(5, 2, counter)
        ax.plot(df[feature], linewidth=0.5, color='red')
        ax.scatter(buy_dates, buys, marker='^', color='darkgreen')
        ax.scatter(sell_dates, sells, marker='v', color='darkred')#, markersize=2)
        ax.axhline(y=stdev_threshold, linestyle='--', color='gray', linewidth=1)
        ax.axhline(y=-stdev_threshold, linestyle='--', color='gray', linewidth=1)
        ax.fill_between(df.index, df[feature], stdev_threshold,
                        where = df[feature] > stdev_threshold,
                        facecolor='crimson', interpolate=True, alpha=0.5)
        ax.fill_between(df.index, df[feature], -stdev_threshold,
                        where = df[feature] < -stdev_threshold,
                        facecolor='green', interpolate=True, alpha=0.5)
        ax.set(title = feature+' Standardized', ylabel='Standard Deviation')
        counter += 1
    plt.show()

    return None

#stock = 'BBCA_IJ'
#plot_stats(stock)
#plot_features(stock)

#df = get_data(stock)
#df_stats = combine_stats(stock, stdev_threshold, window)
#df_stats = df_stats.loc[:, ::2]
#df_stats = df_stats.reindex(sorted(df_stats.columns), axis=1)















