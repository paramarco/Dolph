#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform
securities = []
securities = [
    {   # Calibrated 2026-02-17 | profit score: 22541.24 -> 57050.96 (30d/15K, qty*margin, $2 cost)
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 16,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.195,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 23,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 26,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 57,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.95,
            'MARGIN_EXPANSION_MIN': 0.002429,
            'MARGIN_EXPANSION_MAX': 0.008343,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.004714,
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
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.393,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 13,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.003486,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {   # Calibrated 2026-02-17 | profit score: 43288.95 -> 119297.20 (30d/15K, qty*margin, $2 cost)
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {   # Calibrated 2026-02-17 | profit score: 29245.35 -> 70392.57 (30d/15K, qty*margin, $2 cost)
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 17,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.607,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 8,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 24,
            'BB_STD': 2,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 36,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 57,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0104,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 1.75,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 1.914,
            'DIVERGENCE_LOOKBACK': 11,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.0052,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
    {
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
            'positionMargin': 0.00245,
            'stopLossCoefficient': 26,
            'period': '1Min',
            # VCP
            'VCP_ATR_SLOPE_EXPANSION': 0.169286,
            'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.35,
            'VCP_ADX_TREND_THRESHOLD': 18,
            # Indicator Periods
            'EMA_FAST': 12,
            'EMA_MID': 15,
            'EMA_SLOW': 35,
            'RSI_PERIOD': 18,
            'ATR_PERIOD': 18,
            'ATR_SLOPE_WINDOW': 4,
            'ADX_PERIOD': 10,
            'BB_WINDOW': 14,
            'BB_STD': 1,
            'BB_PERCENTILE_WINDOW': 70,
            'FVP_WINDOW': 39,
            # Expansion Phase Thresholds
            'EXPANSION_DEVIATION_THRESHOLD': 0.00035,
            'EXPANSION_RSI_SHORT_MIN': 28,
            'EXPANSION_RSI_LONG_MAX': 52,
            # Trend Phase Thresholds
            'TREND_RSI_LONG_MIN': 28,
            'TREND_RSI_LONG_MAX': 91,
            'TREND_RSI_SHORT_MIN': 21,
            'TREND_RSI_SHORT_MAX': 63,
            # Margin Adaptation Parameters
            'MARGIN_CONTRACTION_FIXED': 0.00105,
            'MARGIN_EXPANSION_MULTIPLIER': 1.564,
            'MARGIN_EXPANSION_MIN': 0.0014,
            'MARGIN_EXPANSION_MAX': 0.0056,
            'MARGIN_TREND_ATR_MULTIPLIER': 2.6,
            'MARGIN_TREND_MIN': 0.0026,
            'MARGIN_TREND_MAX': 0.0078,
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
            'VOLUME_AVG_WINDOW': 16,
            'VOLUME_SLOPE_WINDOW': 4,
            'BIG_VOLUME_THRESHOLD': 2.34,
            'EXTREME_VOLUME_THRESHOLD': 2.821,
            'BIG_BODY_ATR_THRESHOLD': 0.84,
            'EXTREME_BODY_ATR_THRESHOLD': 2.6,
            'DIVERGENCE_LOOKBACK': 12,
            # Buying Climax
            'BUYING_CLIMAX_LOOKBACK': 14,
            'BUYING_CLIMAX_TREND_LOOKBACK': 10,
            'BUYING_CLIMAX_EXTENSION': 0.004514,
            'BUYING_CLIMAX_COOLDOWN_SECONDS': 900,
            # Final Decision Scoring
            'MIN_TOTAL_SCORE': 1.05,
            'MIN_CONFIDENCE': 0.42,
        }
    },
]
#logLevel = logging.DEBUG
logLevel = logging.INFO
MODE = 'OPERATIONAL' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
periods = ['1Min'] #periods = ['1Min','30Min']

current_tz = pytz.timezone('America/New_York')
# Localize the 'since' and 'until' datetime objects to the specified timezone
since = current_tz.localize(dt.datetime(year=2024, month=12, day=11, hour=10, minute=0))
until = current_tz.localize(dt.datetime(year=2026, month=2, day=12, hour=10, minute=0))
#until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
tradingTimes = (dt.time(9, 34), dt.time(15, 45))


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
