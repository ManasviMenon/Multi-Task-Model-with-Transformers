"""Data loading, feature engineering, and sequence windowing.

`load_and_engineer_features` and `make_sequences` are moved verbatim from
scripts/walk_forward.py - the NaN-drop cascade, the lag chain, and the ordering of the
rolling windows all affect which rows survive (4705 of 4783 raw trading days), so nothing
here is rewritten or "tidied".
"""
import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 5

FEATURES = [
    'log_open_ret', 'log_high_ret', 'log_low_ret', 'log_close_ret',
    'log_volume_lag1', 'log_volume_lag5',
    'ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag4', 'ret_lag5',
    'yz_vol_lag1', 'VaR_5_lag1', 'drawdown', 'ret_min20', 'ret_max20',
    'CVaR_5_lag1', 'ret_skew20'
]
TARGETS = ['log_return', 'log_volume', 'yz_vol', 'VaR_5', 'CVaR_5']

DEFAULT_CSV = "SPX500-18Y_scaled_y.csv"


def load_and_engineer_features(csv_path):
    """Read the OHLCV csv and engineer the feature/target columns.

    The two lines before `engineer_features` are specific to this csv's layout (a trailing
    block of empty columns, and a first row that isn't data). Serving takes the
    `engineer_features` path directly with an in-memory frame instead.
    """
    data = pd.read_csv(csv_path, parse_dates=['observation_date'], dayfirst=True)
    data = data.dropna(axis=1, how='all')
    data = data.drop(index=0)
    return engineer_features(data)


def engineer_features(data):
    """The feature/target engineering itself, on an OHLCV frame already loaded.

    Moved out of `load_and_engineer_features` verbatim so inference can build features from
    request payloads through the identical code path - if serving engineered features even
    slightly differently from training, every prediction would be quietly wrong. A test
    asserts both paths produce byte-identical output on the project csv.

    The NaN-drop cascade is why inference needs a long history: the rolling(20) terms and the
    lag chain consume roughly 78 leading rows before the first usable row appears.
    """
    data = data.copy()

    data['ret'] = data['Adj Close'].pct_change(fill_method=None)
    data['log_return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['log_volume'] = np.log(data['Volume'])

    data['log_open_ret'] = np.log(data['Open'] / data['Adj Close'].shift(1))
    data['log_high_ret'] = np.log(data['High'] / data['Adj Close'].shift(1))
    data['log_low_ret'] = np.log(data['Low'] / data['Adj Close'].shift(1))
    data['log_close_ret'] = np.log(data['Close'] / data['Adj Close'].shift(1))

    data['log_volume_lag1'] = data['log_volume'].shift(1)
    data['log_volume_lag5'] = data['log_volume'].shift(5)

    data['ret_skew20'] = data['log_return'].rolling(window=20).skew()
    data.dropna(inplace=True)

    def yang_zhang_vol(df):
        log_oc = np.log(df['Open'] / df['Adj Close'].shift(1))
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        rs = 0.5 * (log_ho ** 2 + log_lo ** 2)
        return (log_oc ** 2 + rs).rolling(window=20).mean()

    data['yz_vol'] = yang_zhang_vol(data)
    data.dropna(inplace=True)

    data['VaR_5'] = data['log_return'].ewm(span=10).mean() - 1.2 * data['yz_vol']
    data.dropna(inplace=True)

    for lag in range(1, 6):
        data[f'ret_lag{lag}'] = data['log_return'].shift(lag)
    data['yz_vol_lag1'] = data['yz_vol'].shift(1)
    data['VaR_5_lag1'] = data['VaR_5'].shift(1)
    data['CVaR_5'] = data['log_return'].rolling(window=20).apply(lambda x: x[x < x.quantile(0.05)].mean())
    data.dropna(inplace=True)

    data['drawdown'] = (data['Adj Close'] / data['Adj Close'].cummax()) - 1
    data['ret_min20'] = data['log_return'].rolling(20).min()
    data['ret_max20'] = data['log_return'].rolling(20).max()
    data['CVaR_5_lag1'] = data['CVaR_5'].shift(1)
    data.dropna(inplace=True)

    return data.reset_index(drop=True)


def make_sequences(X_scaled, seq_len):
    return np.array([X_scaled[i - seq_len + 1:i + 1] for i in range(seq_len - 1, len(X_scaled))])
