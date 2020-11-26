## Technical_Signals
### Evaluating the signal strength of technical indicators for stocks.

This code provides a way to evaluate the signal strength of a variety of oscillating technical indicators. This example only uses various durations of RSI, moving average spreads, PE/PB and bollinger band %. All signals have been standardized so that outliers can be easily defined as signals above/below a chosen standard deviation threshold.

The results show that not all technical indicators provide the same signal strength for all stocks. For example, an RSI outlier signalling a buy on one stock may be more reliable than the same RSI generating a buy signal for another stock (intuitively this makes sense). 

Therefore, this framework provides an efficient method to assess a multitude of technical signals across a universe of stocks, identify the signal strength for each, and __comb a universe of stocks to find current technical outliers to indentify stocks to consider for investment.__ The following example shows results and analysis for Bank Central Asia (BBCA) in Indonesia.


### Here are the 10 technical indicators evaluated, showing buy/sell signals for a +/-1.5 stdev threshold:

![My image](https://github.com/footfalcon/Technical_Signals/blob/master/plot_features.png)

### And a summary of the results plotted by accuracy:

![My image](https://github.com/footfalcon/Technical_Signals/blob/master/plot_stats.png)

The accuracy percentage scale is re-centred so that 50% accuracy is zero to make it easier to see which signals are above or below 50% accuracy. They are then ranked by descending accuracy. The table below shows some stats to highlight that accuracy alone should not be the determining ranking factor. For example, the table summarizes the top 5 signals in the 1-Quarter Return Outcomes plot. However, PE1_STD_buy is ranked second after PB1_STD_buy in the plot, but since PE1 has both greater frequency and higher return, its score is higher. Similarly, RSI_42_buy is ranked 5th by accuracy alone, but with higher frequency and returns considered, it scores as 2nd most reliable indicator.

![My image](https://github.com/footfalcon/Technical_Signals/blob/master/table.png)

#### Other things to note:
Almost all the indicators with >50% accuracy are buy signals, and vice versa. This would suggest that BBCA_IJ was an upward trending stock and simply buying the dips would have been a good starting point (indeed the price chart confirms this). Still, it is extremely helpful to know which signal to rely upon when buying dips, and PE stands out as by far the most reliable for this stock. The lowest min return of any PE buy signal for any return outcome period was -0.7%, while the lowest max of any period was 15%, implying a very good risk / reward skew.

Interesting also to note that PE_sell scores 5th for the 1-Quarter Return Outcomes, suggesting it is reasonably good as a two-way indicator.

The scoring method is hastily cobbled together and could likely be improved, but intuitively makes sense for ranking. 

Its difficult to avoid duplicate counting of signals, particularly with faster signals, as the level can dip back and forth across the threshold. This has largely been avoided for the 1-week and 1-month return horizons, but is more challenging to deal with for the 1-quarter outcome horizon.

#### Next steps:
With this framework, it is possible to include both a larger universe of stocks and test/evaluate many more technical indicators. There may be interesting findings on which signals are generally good for any stock, which may be good for specific sectors, quality, and which may simply be idiosyncratic to a particular stock. ML clustering could also be considered here.

Finding an ensemble of signals may prove to increase the signal score. Signal frequency drops considerably with more indicators required to be in outlier territory, which in turn can impinge on the reliability of accuracy scores. One possible mitigant worth exploring might be to increase the speed/sensitivity of the indicators such that they become akin to weak learners in a random forest type ensemble, where once combined together they provide a much more robust signal.

Findings here could seen as a form of feature engineering / selection to then use as features in ML models.

With signal scoring, it has been shown that current outliers can be assessed and ranked by signal score to comb the universe for the best potential buys/sells. Proper backtesting needs to be done to better evaluate this as a stock selection methodolgy.


