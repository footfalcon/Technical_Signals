# --- Evaluating Technical Signals -------------------------------------------
''' Get top/bottom outlier stats on all columns (features), eg: features values
    2-sd, get dates, count how many times that feature has reached/exceeded
    such levels, what were the subsequent 1 week/month/qtr returns. '''

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# -- Set up data -------------------------------------------------------------
# Stock data
def get_data(features):
    fpath = '/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/stocks_features_standard/'
    fname = 'CPN_TB.pkl'
    fh = os.path.join(fpath, fname)
    df = pd.read_pickle(fh)
    #features = ['CLOSE', 'LOG_D', 'ADVT', '20D_VOL', 'BBAND_PCT', 'RSI_14', 'MA_50_vs_100', 'PE1']
    df = df[features]
    df.dropna(inplace=True)
    return df

features = ['CLOSE', 'BBAND_PCT_STD', 'RSI_14', 'RSI_28', 'RSI_42', 'RSI_56',
            'MA_5_vs_20','MA_20_vs_50', 'MA_50_vs_100', 'PE1_STD', 'PB1_STD']
df = get_data(features)


# -- Wrap into functions and loop through all features -----------------------
# Global variables
stdev_threshold = 1
window = 5

def get_signal(df, feature, stdev_threshold, window):
    ''' Standardize feature and generate signal '''
    feat_stdize =  (df[feature] - df[feature].mean()) / df[feature].std()
    feat_signal = (feat_stdize > stdev_threshold).astype(int)
    feat_signal = feat_signal - (feat_stdize < -stdev_threshold).astype(int)
    # Get clean signal
    clean_signal = np.zeros((len(df), 1))
    for i in range(len(df)):
        if (sum(feat_signal[i+1 : i+window+1]) >= window) and (sum(feat_signal[i:i+window]) < window):
            clean_signal[i+window] = 1
        if (sum(feat_signal[i+1 : i+window+1]) <= -window) and (sum(feat_signal[i:i+window]) > -window):
            clean_signal[i+window] = -1

    return clean_signal
#clean_signal = get_signal(df, 'RSI_14', stdev_threshold, window)


def get_stats(df, feature, stdev_threshold, window):
    ''' Calculate week/month/quarter outcome returns, hit rate, accuracy etc. '''

    clean_signal = get_signal(df, feature, stdev_threshold, window)
    index = ['Week', 'Month', 'Quarter']  #level_1 index
    cols = ['Freq', 'Hit Rate', 'Accuracy', 'Exp_Rtn', 'Mean_Rtn', 'Max', 'Min', 'Stdev']

    # Buy signal returns
    l = []
    idx_buys = np.where(clean_signal == -1)[0]  # buy when feature is oversold == -1
    for days in [5, 21, 63]:
        rtn = [ (df.CLOSE[i+days] / df.CLOSE[i] * 100 - 100) for i in idx_buys if i+days < len(df) ]
        freq = len(idx_buys)
        hit = np.sum(np.array(rtn) > 0)
        accuracy = hit / freq
        hi = max(rtn)
        low = min(rtn)
        avg_rtn = np.mean(rtn)
        exp_rtn = accuracy * avg_rtn
        stdev = np.std(rtn)
        l.append([freq, hit, accuracy, exp_rtn, avg_rtn, hi, low, stdev])
    buy_stats = pd.DataFrame(np.asarray(l), index=index, columns=cols)
    buy_stats = buy_stats.stack()  # Make into multi-index
    buy_stats = buy_stats.to_frame(feature+'_buy')  # Convert from series to df and name col

    # Sell signal returns
    l = []
    idx_sells = np.where(clean_signal == 1)[0]  # sell when feature is overbought == 1
    for days in [5, 21, 63]:
        rtn = [ (df.CLOSE[i+days] / df.CLOSE[i] * 100 - 100) for i in idx_sells if i+days < len(df) ]
        freq = len(idx_sells)
        hit = np.sum(np.array(rtn) < 0)
        accuracy = hit / freq
        hi = max(rtn)
        low = min(rtn)
        avg_rtn = np.mean(rtn)
        exp_rtn = accuracy * avg_rtn
        stdev = np.std(rtn)
        l.append([freq, hit, accuracy, exp_rtn, avg_rtn, hi, low, stdev])
    sell_stats = pd.DataFrame(np.asarray(l), index=index, columns=cols)
    sell_stats = sell_stats.stack()  # Make into multi-index
    sell_stats = sell_stats.to_frame(feature+'_sell')  # Convert from series to df and name col

    return buy_stats, sell_stats
#buy_signals, sell_signals = get_stats(df, 'RSI_14', stdev_threshold, window)


def combine_features(df, stdev_threshold, window):
    ''' Concat buy/sell signal df's and name cols (works but may not be most concise/efficient) '''
    features = list(df.columns)
    df_final = pd.DataFrame()
    for f in features[1:]:
        print(f)
        buy_signals, sell_signals = get_stats(df, f, stdev_threshold, window)
        df_final = pd.concat([df_final, buy_signals, sell_signals], axis=1)

    return df_final
df_final = combine_features(df, stdev_threshold, window)



# Accuracy/return plots; add subplots for diff Rtn_Pd's?
# Setup each Rtn_Pd df
def get_rtn_pd_df(df_final):
    rtn_pd = ['Week', 'Month', 'Quarter']
    rtn_pd_dict = {}

    df_summary = df_final.loc[(slice(None), ['Accuracy', 'Mean_Rtn', 'Exp_Rtn']), :]
    flat_df = df_summary.reset_index(level=1, drop=False)
    flat_df = flat_df.rename(columns={'level_1':'Stats'})
    flat_df.index.name = 'Rtn_Pd'
    flat_df_T = flat_df.T

    for prd in rtn_pd:
        df_prd = flat_df_T.loc[:, prd]
        df_prd.columns = ['Accuracy', 'Exp_Rtn', 'Mean_Rtn']
        df_prd.name = prd
        df_prd = df_prd[1:]
        df_prd['Accuracy'] = df_prd.Accuracy - 0.5
        df_prd = df_prd.sort_values(by='Accuracy', ascending=False)
        rtn_pd_dict[prd+'_df'] = df_prd

    return rtn_pd_dict
rtn_pd_dict = get_rtn_pd_df(df_final)


def plot_stats(rtn_pd_dict):
    plt.style.use('ggplot')
    plt.figure(figsize=(18, 6), tight_layout=True)
    plt.rc('font', size=6)

    for i in range(len(rtn_pd_dict)):
        name = list(rtn_pd_dict.keys())[i]
        df = rtn_pd_dict[name]

        plt.subplot(1,3, i+1)
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

    return plt.show()
plot_stats(rtn_pd_dict)


def plot_features(df):
    fig = plt.figure(figsize=(18, 16), tight_layout=True)
    features = df.columns[1:-1]
    counter = 1
    for feature in features:
        clean_signal = get_signal(df, feature, stdev_threshold, window)
        idx_buys = np.where(clean_signal == -1)[0]  # buy when feature is oversold == -1
        buys = [ df[feature][i] for i in idx_buys ]
        buy_dates = [ df.index[i] for i in idx_buys ]
        idx_sells = np.where(clean_signal == 1)[0]  # buy when feature is overbought == 1
        sells = [ df[feature][i] for i in idx_sells ]
        sell_dates = [ df.index[i] for i in idx_sells ]

        plot_loc = 520 + counter
        ax = fig.add_subplot(plot_loc)
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

    return plt.show()
plot_features(df)




# --- Dashboard/ref part -----------------------------------------------------
def get_universe():
    fh='/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/universe.pkl'
    df_universe = pd.read_pickle(fh)

    return df_universe
df_universe = get_universe()


def get_current_levels(df_universe, features):
    stocks = df_universe.index.levels[1]
    current_df = pd.DataFrame(columns=features[1:])
    for stock in stocks:
        current = df_universe.loc[(df_universe.index.levels[0][-1], stock), features[1:]]
        current = pd.DataFrame(current).T.reset_index(level=0, drop=True)
        current_df = current_df.append(current)

    return current_df
current_df = get_current_levels(df_universe, features)


def get_outliers(current_df, threshold):
    outlier_buys = current_df[current_df < -threshold].dropna(how='all')
    outlier_sells = current_df[current_df > threshold].dropna(how='all')

    return outlier_buys, outlier_sells
outlier_buys, outlier_sells = get_outliers(current_df, threshold=0.8)









