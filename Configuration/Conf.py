#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging


securities = []
securities = [
    {'board': 'EQTY', 'seccode': 'AAPL'}
    , {'board': 'EQTY', 'seccode': 'INTC'}
    , {'board': 'EQTY', 'seccode': 'NVDA'}
    , {'board': 'EQTY', 'seccode': 'ONCO'}
    # {'board': 'EQTY', 'seccode': 'VERB'},
    # {'board': 'EQTY', 'seccode': 'ONCO'},
    # {'board': 'EQTY', 'seccode': 'KAPA'},
    # {'board': 'EQTY', 'seccode': 'PG'},
    # {'board': 'EQTY', 'seccode': 'KO'}
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
nogoTradingHours = [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,22,23]

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

statusOrderForwarding = ['watching', 'active', 'forwarding', 'new', 'pending_new', 'accepted', 'tp_guardtime', 'tp_forwarding', 'sl_forwarding', 'sl_guardtime' ]
statusOrderExecuted = ['tp_executed', 'sl_executed','filled','matched']
statusOrderCanceled = ['cancelled', 'denied', 'disabled', 'expired', 'failed', 'rejected', 'canceled', 'removed', 'done_for_day']
statusOrderOthers = ["linkwait","tp_correction","tp_correction_guardtime","none","inactive","wait","disabled","failed","refused","pending_cancel", "pending_replace", "stopped", "suspended", "calculated" ]
statusStopOrderExecuted = ['tp_executed', 'sl_executed','matched','triggered']
statusStopOrderFilled = ['filled']

########### default-fallback values ##########################################
factorPosition_Balance = 0.3
factorMargin_Position  = 0.0035
entryTimeSeconds = 3600
exitTimeSeconds = 36000
stopLossCoefficient = 6
correction = 0.0
spread = 0.0