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
def get_data():
    fpath = '/Users/martinfoot/Library/Mobile Documents/com~apple~CloudDocs/DATA SCIENCE/Trading_Strategies/X - Data/stocks_w_features/'
    fname = 'CPN_TB.pkl'
    fh = os.path.join(fpath, fname)
    df = pd.read_pickle(fh)
    features = ['CLOSE', 'LOG_D', 'ADVT', '20D_VOL', 'BBAND_PCT', 'RSI_14', 'MA_50_vs_100', 'PE1']
    df = df[features]
    df.dropna(inplace=True)
    return df
df = get_data()


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
    # get index for calculating returns
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
    buy_signals = pd.DataFrame(np.asarray(l), index=index, columns=cols)
    buy_signals = buy_signals.stack()  # Make into multi-index
    buy_signals = buy_signals.to_frame(feature+'_buy')  # Convert from series to df and name col

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
    sell_signals = pd.DataFrame(np.asarray(l), index=index, columns=cols)
    sell_signals = sell_signals.stack()  # Make into multi-index
    sell_signals = sell_signals.to_frame(feature+'_sell')  # Convert from series to df and name col

    return buy_signals, sell_signals
#buy_signals, sell_signals = get_stats(df, 'RSI_14', stdev_threshold, window)


def run(df, stdev_threshold, window):
    ''' Concat buy/sell signal df's and name cols (works but may not be most concise/efficient) '''
    features = list(df.columns)
    df_final = pd.DataFrame()
    for f in features[3:]:
        print(f)
        buy_signals, sell_signals = get_stats(df, f, stdev_threshold, window)
        df_final = pd.concat([df_final, buy_signals, sell_signals], axis=1)

    return df_final
df_final = run(df, stdev_threshold, window)



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

    for pd in rtn_pd:
        df_pd = flat_df_T.loc[:, pd]
        df_pd.columns = ['Accuracy', 'Exp_Rtn', 'Mean_Rtn']
        df_pd.name = pd
        df_pd = df_pd[1:]
        df_pd['Accuracy'] = df_pd.Accuracy - 0.5
        df_pd = df_pd.sort_values(by='Accuracy', ascending=False)
        rtn_pd_dict[pd+'_df'] = df_pd

    return rtn_pd_dict
rtn_pd_dict = get_rtn_pd_df(df_final)


def get_plots(rtn_pd_dict):
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
        ax2.plot(df.Exp_Rtn, 'b^')
        ax2.plot(df.Mean_Rtn, 'gs')
        _, ax2yhigh = ax2.get_ylim()  # these 2 lines set second y-axis range to match primary range...
        ax2.set_ylim(ax2yhigh *  ylow / yhigh , ax2yhigh)
        ax2.set_ylabel('Returns', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        plt.legend()

    return plt.show()
get_plots(rtn_pd_dict)










