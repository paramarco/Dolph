#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime as dt
import pytz
import logging


securities = []
securities.append({'board': 'EQTY', 'seccode': 'AAPL'})
# securities.append({'board': 'EQTY', 'seccode': 'ADDYY'}) 
# securities.append({'board': 'EQTY', 'seccode': 'ALIZY'})
# securities.append({'board': 'EQTY', 'seccode': 'BASFY'})
# securities.append({'board': 'EQTY', 'seccode': 'BAYRY'})
# securities.append({'board': 'EQTY', 'seccode': 'BEINY'})
# securities.append({'board': 'EQTY', 'seccode': 'BMWYY'})
# securities.append({'board': 'EQTY', 'seccode': 'DTEGY'})
# securities.append({'board': 'EQTY', 'seccode': 'SAP'})
# securities.append( {'board':'FUT', 'seccode':'GZZ0'} )
# securities.append( {'board':'FUT', 'seccode':'GZZ0'} ) 
# securities.append( {'board':'FUT', 'seccode':'SRZ0'} ) 
#logLevel = logging.DEBUG 
logLevel = logging.INFO
MODE = 'OPERATIONAL' # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL'
periods = ['1Min'] #periods = ['1Min','30Min']

current_tz = pytz.timezone('America/New_York')
# Localize the 'since' and 'until' datetime objects to the specified timezone
since = current_tz.localize(dt.datetime(year=2022, month=8, day=7, hour=7, minute=0))
until = current_tz.localize(dt.datetime(year=2024, month=9, day=18, hour=10, minute=0))
between_time = (
    current_tz.localize(dt.datetime.strptime('07:00', '%H:%M')).time(),
    current_tz.localize(dt.datetime.strptime('23:40', '%H:%M')).time()
)
nogoTradingHours = [7,21]

numTestSample = 100
TrainingHour = 10  # 10:00 
currentTestIndex = 0  

dbname='dolph_db'
user='dolph_user'
password='dolph_password'
host='127.0.0.1'


transaqConnectorPort = 13000
transaqConnectorHost = '127.0.0.1'

statusOrderForwarding = ['watching', 'active', 'forwarding', 'new', 'pending_new', 'accepted', 'tp_guardtime', 'tp_forwarding', 'sl_forwarding', 'sl_guardtime' ]
statusOrderExecuted = ['tp_executed', 'sl_executed','filled','matched']
statusOrderCanceled = ['cancelled', 'denied', 'disabled', 'expired', 'failed', 'rejected', 'canceled', 'removed', 'done_for_day']
statusOrderOthers = ["linkwait","tp_correction","tp_correction_guardtime","none","inactive","wait","disabled","failed","refused","pending_cancel", "pending_replace", "stopped", "suspended", "calculated" ]
statusStopOrderExecuted = ['tp_executed', 'sl_executed','matched','triggered']
statusStopOrderFilled = ['filled']

factorPosition_Balance = 0.3
factorMargin_Position  = 0.005