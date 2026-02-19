#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform
securities = [
    {   # Calibrated 2026-02-18 | profit score: 60740.60 -> 75762.12 (30d/15K, qty*margin, $2 cost)
        'seccode': 'AAPL',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.20335714285714288,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.33499999999999996,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 23,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 12,
            'BB_WINDOW': 16,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 23,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 50,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 59,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 2.033571428571429,
            'MARGIN_EXPANSION_MIN': 0.0027413000000000003,
            'MARGIN_EXPANSION_MAX': 0.0108459,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0031571428571428566,
            'MARGIN_TREND_MAX': 0.004916028571428572,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 12,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 3.1109,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.7114285714285717,
            'DIVERGENCE_LOOKBACK': 9,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0039342,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 120863.20 -> 159413.57 (30d/15K, qty*margin, $2 cost)
        'seccode': 'INTC',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 3,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 25,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.1620308857142857,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 19,
            'ATR_SLOPE_WINDOW': 5,
            'ADX_PERIOD': 13,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 47,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 50,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 95,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.2288571428571429,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00488,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 13,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 2.7001,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 3.157142857142857,
            'DIVERGENCE_LOOKBACK': 10,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.005481285714285714,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 76613.94 -> 115715.02 (30d/15K, qty*margin, $2 cost)
        'seccode': 'NVDA',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 21,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.2055615714285714,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.6850428571428571,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 10,
            'EMA_MID': 12,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 23,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 12,
            'BB_WINDOW': 29,
            'BB_STD': 3,
            'BB_PERCENTILE_WINDOW': 55,
            'FVP_WINDOW': 44,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 50,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 103,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 59,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 2.0332000000000003,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.013519999999999999,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.008802857142857142,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 17,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.8250000000000002,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.4882,
            'DIVERGENCE_LOOKBACK': 8,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0067599999999999995,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 23633.30 -> 31561.43 (30d/15K, qty*margin, $2 cost)
        'seccode': 'SOFI',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.11850019999999999,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 13,
            'EMA_MID': 14,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 13,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 13,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.0010049999999999998,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 11,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.9747,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 1.8199999999999998,
            'DIVERGENCE_LOOKBACK': 8,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.0031597999999999995,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 29754.86 -> 41248.42 (30d/15K, qty*margin, $2 cost)
        'seccode': 'MARA',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.11850019999999999,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 20,
            'ATR_PERIOD': 20,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 13,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 103,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.157142857142857,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 14,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.9747,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 3.157142857142857,
            'DIVERGENCE_LOOKBACK': 8,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 13,
            'BUYING_CLIMAX_EXTENSION': 0.003933628571428571,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 25166.21 -> 43922.03 (30d/15K, qty*margin, $2 cost)
        'seccode': 'RIVN',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 29,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.11850019999999999,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.33499999999999996,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 16,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 7,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 11,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.8385714285714285,
            'EXTREME_VOLUME_THRESHOLD': 3.4255,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.4885714285714284,
            'DIVERGENCE_LOOKBACK': 8,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 8,
            'BUYING_CLIMAX_EXTENSION': 0.0031597999999999995,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 27814.78 -> 36880.27 (30d/15K, qty*margin, $2 cost)
        'seccode': 'HOOD',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 20,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.1620308857142857,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 10,
            'EMA_MID': 12,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 19,
            'ATR_PERIOD': 23,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.7114285714285717,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 13,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 3.6673000000000004,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.265714285714286,
            'DIVERGENCE_LOOKBACK': 11,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0035467142857142854,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 22924.76 -> 28493.67 (30d/15K, qty*margin, $2 cost)
        'seccode': 'SMCI',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.11850019999999999,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 13,
            'EMA_MID': 14,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 22,
            'ATR_PERIOD': 14,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 95,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.0012749999999999999,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.002934285714285714,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 15,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.9747,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 1.8199999999999998,
            'DIVERGENCE_LOOKBACK': 11,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0031597999999999995,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 27762.85 -> 35060.14 (30d/15K, qty*margin, $2 cost)
        'seccode': 'DKNG',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.13301042857142856,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 22,
            'ATR_PERIOD': 13,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 12,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 27,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 32,
            'TREND_RSI_LONG_MAX': 103,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.0948,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00392,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0031571428571428566,
            'MARGIN_TREND_MAX': 0.008134285714285714,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 13,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.9747,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.0428571428571427,
            'DIVERGENCE_LOOKBACK': 11,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0031597999999999995,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 29560.45 -> 41287.09 (30d/15K, qty*margin, $2 cost)
        'seccode': 'MSTR',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.1620308857142857,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 10,
            'EMA_MID': 12,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 22,
            'ATR_PERIOD': 23,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 11,
            'BB_WINDOW': 18,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 44,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 36,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 2.0332000000000003,
            'MARGIN_EXPANSION_MIN': 0.00098,
            'MARGIN_EXPANSION_MAX': 0.00728,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.01014,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 14,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 3.6673000000000004,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 1.8199999999999998,
            'DIVERGENCE_LOOKBACK': 8,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0058682,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 15801.48 -> 20599.87 (30d/15K, qty*margin, $2 cost)
        'seccode': 'AMZN',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.14752065714285714,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 23,
            'ATR_PERIOD': 17,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 13,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 49,
            'FVP_WINDOW': 41,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 50,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 87,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.0007349999999999999,
            'MARGIN_EXPANSION_MULTIPLIER': 2.0332000000000003,
            'MARGIN_EXPANSION_MIN': 0.00182,
            'MARGIN_EXPANSION_MAX': 0.00728,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.0033799999999999998,
            'MARGIN_TREND_MAX': 0.008134285714285714,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 11,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 3.1837,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 2.4885714285714284,
            'DIVERGENCE_LOOKBACK': 10,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.0058682,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
    {   # Calibrated 2026-02-18 | profit score: 10913.76 -> 15615.60 (30d/15K, qty*margin, $2 cost)
        'seccode': 'MSFT',
        'board': 'EQTY',
        'market': 'NASDAQ',
        'decimals': 2,
        'id': 0,
        'params': {
            'algorithm': 'MinerviniClaude',
            'entryByMarket': True,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51,
            'positionMargin': 0.0017149999999999997,
            'stopLossCoefficient': 18,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.19105134285714284,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.24499999999999997,
            'VCP_ADX_TREND_THRESHOLD': 13,
            # Indicator Periods
            'EMA_FAST': 14,
            'EMA_MID': 16,
            'EMA_SLOW': 24,
            'RSI_PERIOD': 19,
            'ATR_PERIOD': 23,
            'ATR_SLOPE_WINDOW': 3,
            'ADX_PERIOD': 13,
            'BB_WINDOW': 10,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 91,
            'FVP_WINDOW': 41,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.000245,
            'EXPANSION_RSI_SHORT_MIN': 20,
            'EXPANSION_RSI_LONG_MAX': 50,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 20,
            'TREND_RSI_LONG_MAX': 95,
            'TREND_RSI_SHORT_MIN': 15,
            'TREND_RSI_SHORT_MAX': 60,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.001365,
            'MARGIN_EXPANSION_MULTIPLIER': 1.6310285714285715,
            'MARGIN_EXPANSION_MIN': 0.00182,
            'MARGIN_EXPANSION_MAX': 0.005359999999999999,
            'MARGIN_TREND_ATR_MULTIPLIER': 3.3800000000000003,
            'MARGIN_TREND_MIN': 0.002934285714285714,
            'MARGIN_TREND_MAX': 0.00546,
            # Calibration Parameters
            'CALIBRATION_LOOKBACK_DAYS': 30,
            'CALIBRATION_LIMIT_RESULTS': 15000,
            'CALIBRATION_MIN_ROWS': 1000,
            'CALIBRATION_MARGIN_MIN': 0.001,
            'CALIBRATION_MARGIN_MAX': 0.006,
            'CALIBRATION_MARGIN_STEPS': 10,
            # Calibration Simulation Parameters
            'CALIBRATION_LOOKAHEAD_BARS': 60,
            'CALIBRATION_STOPLOSS_MULTIPLIER': 5.0,
            'CALIBRATION_DEFAULT_MARGIN': 0.003,
            # Volume Analysis Parameters
            'VOLUME_AVG_WINDOW': 11,
            'VOLUME_SLOPE_WINDOW': 3,
            'BIG_VOLUME_THRESHOLD': 1.638,
            'EXTREME_VOLUME_THRESHOLD': 1.9747,
            'BIG_BODY_ATR_THRESHOLD': 0.588,
            'EXTREME_BODY_ATR_THRESHOLD': 1.8199999999999998,
            'DIVERGENCE_LOOKBACK': 11,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 10,
            'BUYING_CLIMAX_TREND_LOOKBACK': 7,
            'BUYING_CLIMAX_EXTENSION': 0.003933628571428571,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 0.735,
            'MIN_CONFIDENCE': 0.294,
        }
    },
]
#logLevel = logging.DEBUG
logLevel = logging.INFO
MODE = 'OPERATIONAL' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']
numDaysHistCandles = 29

current_tz = pytz.timezone('America/New_York')
# Localize the 'since' and 'until' datetime objects to the specified timezone
since = current_tz.localize(dt.datetime.now() - dt.timedelta(days=numDaysHistCandles))
until = current_tz.localize(dt.datetime(year=2026, month=2, day=18, hour=10, minute=0))
#until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
tradingTimes = (dt.time(9, 44), dt.time(15, 45))


numTestSample = 500
TrainingHour = 10  # 10:00 
currentTestIndex = 0  

db_connection_params = {
    "dbname" : "dolph_db",
    "user" : "dolph_user",
    "password" : "dolph_password",
    "host" : "127.0.0.1",
    "port" : 4713,
    "sslmode" : "disable"    
}

transaqConnectorPort = 13000
transaqConnectorHost = '127.0.0.1'

statusOrderForwarding = ['PendingSubmit', 'Submitted','PendingSubmit',  'watching', 'active', 'forwarding', 'new', 'pending_new', 'accepted', 'tp_guardtime', 'tp_forwarding', 'sl_forwarding', 'sl_guardtime','submitted', 'PreSubmitted', 'inactive']
statusOrderExecuted = ['Filled','tp_executed', 'sl_executed','filled','matched']
statusOrderCanceled = ['Cancelled','cancelled','Rejected','Stopped','denied', 'disabled', 'expired', 'failed', 'rejected', 'canceled', 'removed', 'done_for_day']
statusOrderOthers = ['PartiallyFilled','Inactive','PendingCancel', "linkwait","tp_correction","tp_correction_guardtime","none","inactive","wait","disabled","failed","refused","pending_cancel", "pending_replace", "stopped", "suspended", "calculated" ]
statusExitOrderExecuted = ['tp_executed', 'sl_executed','matched','triggered']
statusExitOrderFilled = ['filled','Filled']

########### default-fallback values ##########################################
factorPosition_Balance = 0.23
factorMargin_Position  = 0.0035
entryTimeSeconds = 3600
exitTimeSeconds = 36000
stopLossCoefficient = 20
correction = 0.0
spread = 0.0

openaikey = platform['secrets']['openaikey']

# ===== Phase Detection Parameters =====
# Expansion:  (ATR_slope > threshold AND Bollinger width percentile high)
# Trend: ( ADX above threshold, strong directional movement) and (EMA alignment either bullish or bearish)
VCP_ATR_SLOPE_EXPANSION = 0.15
VCP_BB_WIDTH_PERCENTILE_EXPANSION = 0.5
VCP_ADX_TREND_THRESHOLD = 25

# Indicator Periods
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50

RSI_PERIOD = 14             # RSI (Momentum Filter). to Confirm directional entries, Filter false breakouts  

ATR_PERIOD = 14             # ATR CALCULATION. to measure volatility level
ATR_SLOPE_WINDOW = 5        # ATR SLOPE. Measures volatility expansion / contraction speed

ADX_PERIOD = 14             # ADX + DI (Trend Strength). to confirm directional strength, to filter ranging markets

BB_WINDOW = 20              # BOLLINGER BAND WIDTH. Measures compression vs expansion of volatility
BB_STD = 2
BB_PERCENTILE_WINDOW = 100

FVP_WINDOW = 30             # FAIR VALUE PRICE (FVP). Rolling mean of close. Used for mean-reversion during expansion

# ===============================
# Expansion Phase Thresholds
# ===============================
EXPANSION_DEVIATION_THRESHOLD = 0.0005
EXPANSION_RSI_SHORT_MIN = 40
EXPANSION_RSI_LONG_MAX = 60

# ===============================
# Trend Phase Thresholds
# ===============================
TREND_RSI_LONG_MIN = 40
TREND_RSI_LONG_MAX = 70

TREND_RSI_SHORT_MIN = 30
TREND_RSI_SHORT_MAX = 60

# ===================================
# Margin Adaptation Parameters
# ===================================

# Contraction Phase
MARGIN_CONTRACTION_FIXED = 0.0015

# Expansion Phase
MARGIN_EXPANSION_MULTIPLIER = 1.5
MARGIN_EXPANSION_MIN = 0.002
MARGIN_EXPANSION_MAX = 0.008

# Trend Phase
MARGIN_TREND_ATR_MULTIPLIER = 2.0
MARGIN_TREND_MIN = 0.002
MARGIN_TREND_MAX = 0.006

# ===================================
# Calibration Parameters
# ===================================

CALIBRATION_LOOKBACK_DAYS = 90
CALIBRATION_LIMIT_RESULTS = 5000
CALIBRATION_MIN_ROWS = 1000

CALIBRATION_MARGIN_MIN = 0.001
CALIBRATION_MARGIN_MAX = 0.006
CALIBRATION_MARGIN_STEPS = 10

# ===================================
# Calibration Simulation Parameters
# ===================================

CALIBRATION_LOOKAHEAD_BARS = 60
CALIBRATION_STOPLOSS_MULTIPLIER = 5.0
CALIBRATION_DEFAULT_MARGIN = 0.003

# ==========================================
# Volume Analysis Parameters
# ==========================================

VOLUME_AVG_WINDOW = 20
VOLUME_SLOPE_WINDOW = 5

BIG_VOLUME_THRESHOLD = 1.8
EXTREME_VOLUME_THRESHOLD = 2.5

BIG_BODY_ATR_THRESHOLD = 1.2
EXTREME_BODY_ATR_THRESHOLD = 2.0

DIVERGENCE_LOOKBACK = 10

# ==========================================
# BUYING_CLIMAX
# ==========================================

BUYING_CLIMAX_LOOKBACK = 20
BUYING_CLIMAX_TREND_LOOKBACK = 15
BUYING_CLIMAX_EXTENSION = 0.004   # 0.4%
BUYING_CLIMAX_COOLDOWN_SECONDS = 900  # 15 minutos

# ==========================================
# FINAL DECISION SCORING
# ==========================================
MIN_TOTAL_SCORE = 1.5
MIN_CONFIDENCE = 0.6
