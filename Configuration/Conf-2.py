#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging
from Configuration import TradingPlatfomSettings as tps

platform = tps.platform
securities = []
securities = [
    {
        'seccode': 'AAPL'
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2 
        ,'id' : 0
        ,'params': {
            'algorithm': 'stochastic_and_rsi',
            'entryByMarket': False,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"longPositionMargin": 0.0035 
            ,"shortPositionMargin": 0.0035 
            ,"stopLossCoefficient": 3 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
        }
    }
    ,{
        'seccode': 'INTC'       
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2
        ,'id' : 0
        ,'params': {
            'algorithm': 'stochastic_and_rsi',
            'entryByMarket': False,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"longPositionMargin": 0.0035 
            ,"shortPositionMargin": 0.0035 
            ,"stopLossCoefficient": 3 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
        }
    }
    ,{
        'seccode': 'NVDA'        
        ,'board': 'EQTY'
        ,'market': 'NASDAQ'
        ,'decimals' : 2
        ,'id' : 0
        ,'params': {
            'algorithm': 'stochastic_and_rsi',
            'entryByMarket': False,
            'exitTimeSeconds': 36000,
            'entryTimeSeconds': 3600,
            'minNumPastSamples': 51
            ,"longPositionMargin": 0.0035 
            ,"shortPositionMargin": 0.0035 
            ,"stopLossCoefficient": 3 
            ,"acceptableTrainingError": 0.000192
            ,'period': '1Min'
        }
    }
]
logLevel = logging.DEBUG 
#logLevel = logging.INFO
#MODE = 'OPERATIONAL' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL' | 'INIT_DB'
MODE = 'OPERATIONAL'
periods = ['1Min'] #periods = ['1Min','30Min']

current_tz = pytz.timezone('America/New_York')
# Localize the 'since' and 'until' datetime objects to the specified timezone
since = current_tz.localize(dt.datetime(year=2023, month=12, day=11, hour=10, minute=0))
until = current_tz.localize(dt.datetime(year=2024, month=11, day=1, hour=10, minute=0))
#until = current_tz.localize(dt.datetime.now())
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
nogoTradingHours = [0,1,2,3,4,5,6,7,8,9,20,21,22,23]

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
statusStopOrderExecuted = ['tp_executed', 'sl_executed','matched','triggered']
statusStopOrderFilled = ['filled','Filled']

########### default-fallback values ##########################################
factorPosition_Balance = 0.3
factorMargin_Position  = 0.0035
entryTimeSeconds = 3600
exitTimeSeconds = 36000
stopLossCoefficient = 3
correction = 0.0
spread = 0.0

