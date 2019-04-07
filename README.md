## Technical_Signals
### Evaluating the signal strength of technical indicators for stocks.

All technical indicators have been standardized and made into oscilators (if not already).
A buy or sell signal is generated when the oscillator breaches a certain level (default +/-1 stdev).
Stats are collected such as frequency of signal, hit rate (accuracy), and return outcomes for 
1 week/month/quarter periods.

NOTE: returns are 'as is', meaning a sell-signal with a negative return outcome is a correct signal.

Code is set up for a single stock, but intent is to be able to evaluate across a universe of stocks. 
