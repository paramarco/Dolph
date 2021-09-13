# -*- coding: utf-8 -*- 
# @author: Dolph investments

import logging
import sys
import signal
import gc; gc.collect()
import datetime as dt
import pytz
import time

import copy
import pandas as pd
                                                      
import DataServer as ds
import TradingPlatform as tp

import peaks_and_valleys as fluctuationModel

class Dolph:
    def __init__(self, securities):
        
        self.securities = securities
        
        # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL'
        self.MODE = 'TEST_ONLINE' 

        self.numTestSample = 100
        self.since = dt.datetime(year=2021,month=1,day=1,hour=9, minute=0)
        self.until = dt.datetime(year=2021,month=1,day=27,hour=22, minute=0)        
        self.between_time = ('07:00', '23:40')
        self.TrainingHour = 10
    
        # if self.MODE == 'TRAIN_OFFLINE' or self.MODE == 'TEST_OFFLINE':
            
        #     if self.TrainingHour in range(9,14):
        #         self.between_time = ('07:00', '17:00')
        #     else:
        #         self.between_time = ('14:00', '23:00')
   
               
        self.periods = ['1Min','10Min']

        self.data = {}
        self.inputDataTest = {}
        self.lastUpdate = None
        self.currentTestIndex = 0  
        
        logFormat = '%(asctime)s | %(levelname)s | %(funcName)s |%(message)s'
        logging.basicConfig(
            #level = logging.INFO , 
            level = logging.DEBUG , 
            format = logFormat,
            handlers=[  
                logging.FileHandler("./log/Dolph.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('running on mode: ' + self.MODE)

        self.ds = ds.DataServer()
        self.getData = None
        self.getTrainingModel = None
        self.showPrediction = None      

        
        for sec in self.securities: 
            sec['params'] = self.ds.getSecurityAlgParams( sec )
            sec['models'] = {}
            sec['predictions'] = {}
            sec['lastPositionTaken'] = None
            sec['savedEntryPrice'] = None
            for p in self.periods:
                sec['predictions'][p] = []         
               
        self.target = securities[0]
        self.params = self.ds.getSecurityAlgParams(self.target)
        self.minNumPastSamples = self.params['minNumPastSamples']  

        alg = self.params['algorithm']  
        
       
        if (alg == 'peaks_and_valleys' ):
            self.getData = self.ds.searchData
            self.getTrainingModel = fluctuationModel.Model      
        else:
            raise RuntimeError('algorithm not found')

        connectOnInit = False
        if (self.MODE == 'TEST_ONLINE' or self.MODE == 'OPERATIONAL' ):
            connectOnInit = True
            
        self.tp = tp.TradingPlatform(
            self.target, 
            self.securities, 
            self.onHistoryCandleRes,
            connectOnInit,
            self.onCounterPosition
        )
        
        def signalHandler(signum, frame):
            self.tp.disconnect()
            print ('hasta la vista!')
            sys.exit()
        
        signal.signal(signal.SIGINT, signalHandler)
        
        self.goodLuckStreak  = 100

    def onHistoryCandleRes(self, obj):
        logging.debug( repr(obj) )            
        self.ds.storeCandles(obj)
     
        
    def getSecurityBySeccode(self, seccode):
        sec = None
        for s in self.securities: 
            if seccode == s['seccode']:
                sec = s
                break
        return sec
        
    def getLastClosePrice(self, seccode):
        
        sec = self.getSecurityBySeccode( seccode )
        periods = self.periods
        since = dt.datetime.now() - dt.timedelta( hours = 12 ) 
        dfs = {}
        
        if self.MODE == 'TEST_OFFLINE' :
            dfs = self.getData(
                self.securities, periods, self.since, sec, self.between_time, 
                self.until
            )   
        else:
            dfs = self.getData(
                self.securities, periods, since, sec, self.between_time
            )   

        dataFrame_1min = dfs['1Min']
        df_1min = dataFrame_1min[ dataFrame_1min['Mnemonic'] == seccode ]
        timelast1Min = df_1min.tail(1).index
        timelast1Min = timelast1Min.to_pydatetime()
        LastClosePrice = df_1min['EndPrice'].iloc[-1]
        logging.info(' time last 1Min: ' + str(timelast1Min) )
        logging.info(' Close last 1Min: ' + str(LastClosePrice) )

        return LastClosePrice
        
    def onCounterPosition(self, position2invert ):
        
        position = copy.deepcopy(position2invert)
        if (position2invert.takePosition == "long"):           
            position.takePosition = "short"
        elif (position2invert.takePosition  == "short"):
            position.takePosition = "long"
        else:
            logging.error( "takePosition must be either long or short")
            raise Exception( position.takePosition )
        
        
        sec = self.getSecurityBySeccode( position.seccode )
        if sec is None: logging.error('sec not found... '); return;
        
        lastClosePrice = self.getLastClosePrice(position.seccode)
        if lastClosePrice is None :  logging.error('for seccode...'); return;
        
        
        params = self.ds.getSecurityAlgParams( sec )       
        entryTimeSeconds =      params['entryTimeSeconds']
        exitTimeSeconds =       params['exitTimeSeconds'] 
        k =                     params['stopLossCoefficient']
        marginsByHour =         params['positionMargin']
        correctionByHour =      params['correction']
        spreadByHour =          params['spread']
        
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = dt.datetime.now(moscowTimeZone)
        moscowHour = moscowTime.hour        
        
        spread =        spreadByHour[str(moscowHour)]
        correction =    correctionByHour[str(moscowHour)]
        deltaForExit =  marginsByHour[str(moscowHour)]

        position.exitTime = moscowTime + dt.timedelta(seconds = exitTimeSeconds)
        smallDeltaExtreamCase = params['smallDeltaExtreamCase']
        
        if position.takePosition == 'long':
            position.entryPrice = lastClosePrice + smallDeltaExtreamCase
            position.exitPrice =position.entryPrice+deltaForExit
            position.stoploss = position.entryPrice  - k * deltaForExit
            
        elif position.takePosition == 'short':
            position.entryPrice = lastClosePrice - smallDeltaExtreamCase
            position.exitPrice =position.entryPrice-deltaForExit
            position.stoploss = position.entryPrice  + k * deltaForExit
        else:
            logging.info('this shouldnt happen ' + position.takePosition )

        
        logging.info('sending '+position.takePosition+' to Trading platform')
        self.tp.processPosition(position)     
       
    
           
    def isSufficientData (self, dataFrame):
        
        # periods = self.periods
        # longestPeriod = periods[-1]
        # dataFrame = dataFrame[longestPeriod]
        dataFrame = dataFrame['1Min']
        
        msg = 'there is only %s samples now, you need at least %s samples for '
        msg+= 'the model to be able to predict'
        sufficient = True
        for sec in self.securities:
            seccode = sec['seccode']
            df = dataFrame[ dataFrame['Mnemonic'] == seccode ]
            numSamplesNow = len(df.index)
            if ( numSamplesNow < self.minNumPastSamples ):
                logging.warning(msg, numSamplesNow,self.minNumPastSamples)            
                sufficient = False
                break
            
        return sufficient
     
    def isPeriodSynced(self, dfs):
        
        synced = False
        # periods = self.periods
        # period = periods[-1]        
        # numPeriod = int(period[0])
        # dataFrame = dfs[period]
        numPeriod = 1
        dataFrame = dfs['1Min']
        dataFrame_1min = dfs['1Min']
        
        for sec in self.securities:
            seccode = sec['seccode']
            df = dataFrame[ dataFrame['Mnemonic'] == seccode ]
            df_1min = dataFrame_1min[ dataFrame_1min['Mnemonic'] == seccode ]
        
        
            timelastPeriod = df.tail(1).index
            timelastPeriod = timelastPeriod.to_pydatetime()
     
            timelast1Min = df_1min.tail(1).index
            timelast1Min = timelast1Min.to_pydatetime()
            
            nMin = -numPeriod + 1
            timeAux = timelast1Min + dt.timedelta(minutes = nMin)
            
            if (timeAux >= timelastPeriod and self.lastUpdate != timelastPeriod):
                synced = True
                self.lastUpdate = timelastPeriod
        
            logging.debug(' timelastPeriod: '  + str(timelastPeriod) )
            logging.debug(' timelast1Min: '     + str(timelast1Min) )
            logging.debug(' timeAux: '     + str(timeAux) )
            logging.debug(' period synced: ' + str(synced) )

        
        return synced

    def dataAcquisition(_):
        
        _.since = _.since +  dt.timedelta( minutes = 1 ) 
        since = _.since 
        _.until = _.since +  dt.timedelta( minutes = _.numTestSample ) 
        until = _.until 
        
        periods = _.periods
        securities = _.securities
        dfs = {}
        
        if (_.MODE == 'TRAIN_OFFLINE'):

            dfs = _.getData(securities, periods, since, None, _.between_time)
            for p in periods:
                _.data[p] =  dfs[p]                

        elif (_.MODE == 'TEST_OFFLINE'):
            
            _.data = \
            _.getData(securities, periods, since, None, _.between_time, until )
            
            
        elif ( _.MODE == 'TEST_ONLINE' or _.MODE == 'OPERATIONAL'):   
            
            #since = dt.date.today() - dt.timedelta(days=3)            
            since =  dt.datetime.now() - dt.timedelta( hours = 24 ) 
            while True:    
                dfs = _.getData(securities, periods, since, None, _.between_time )
                if not _.isSufficientData(dfs) :
                    continue
                if _.isPeriodSynced(dfs):
                    break
                time.sleep(1.5) 
                
            for p in periods:
                _.data[p] = dfs[p]            
        
        else:
            _.log.error('wrong running mode, check self.MODE')


    def trainModel(self, sec, period, params, hour):
        
        msg = 'training&prediction params: '+ period +' '+ str(params)
        logging.info( msg )        
        
        sec['models'][period] = self.getTrainingModel(
            self.data[period], 
            params, 
            period,
            self.MODE,
            hour
        )

    
    def storePrediction(self, sec, prediction, period, params):

        sec['predictions'][period].append(prediction)
   
    def loadModel(self, sec, p , params):        
     
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = dt.datetime.now(moscowTimeZone)
        currentHour = moscowTime.hour
    
        if self.MODE == 'TRAIN_OFFLINE' or self.MODE == 'TEST_OFFLINE':
            currentHour = self.TrainingHour

        if p not in sec['models']:
            self.trainModel(sec, p, params, currentHour )
        elif not sec['models'][p].isSuitableForThisTime(currentHour):
            self.trainModel(sec, p, params, currentHour)
        else:
            logging.debug('model already loaded') 


    def predict( self ):
        
        for p in self.periods:
            for sec in self.securities:
                params = self.ds.getSecurityAlgParams( sec )
                self.loadModel(sec, p, params)            
                logging.debug( 'calling the model ...')
                seccode = sec['seccode']
                dataframe = self.data[p]
                df = dataframe[ dataframe['Mnemonic'] == seccode ]
                pred = sec['models'][p].predict( df, sec,p )            
                self.storePrediction( sec, pred, p, params)
        
   
    def displayPredictions (self):
        
        #logging.info( 'plotting a graph ...') 
        p = self.periods[-1]
        
        for sec in self.securities:
            preds = copy.deepcopy(sec['predictions'][p])
            # self.showPrediction( preds , p)    
      
    
    

    def reviewForHigherfrequency (self, statusLowFreq, sec ):
        
        periodHighFreq = self.periods[0]
        periodLowFreq = self.periods[-1]
        numPeriodLowFreq    =   int(periodLowFreq[0])
        numPeriodHighFreq   =   int(periodHighFreq[0])
        periodGap = numPeriodLowFreq - numPeriodHighFreq  
        predictions = copy.deepcopy(sec['predictions'][periodHighFreq])                
        fluctuation = predictions[-1]
        numWindowSize = fluctuation['samplingWindow'].shape[0]
        indexPenultimate = numWindowSize - 2
        status = 0
      
        indexLastPeak = fluctuation['peak_idx'][-1] if fluctuation['peak_idx'].any() else 0
        indexLastValley = fluctuation['valley_idx'][-1] if fluctuation['valley_idx'].any() else 0
            
        if statusLowFreq == 1 and indexLastPeak > indexLastValley and \
            abs( indexPenultimate - indexLastPeak ) <= periodGap :            
            status = 1
        elif statusLowFreq == -1 and indexLastValley > indexLastPeak and \
            abs( indexPenultimate - indexLastValley ) <= periodGap:
            status = -1
        else: # statusLowFreq == 0
            status = 0 
        
        return status
        

    def evaluatePosition (self, security):        

        longestPeriod = self.periods[-1]
        board = security['board']
        seccode = security['seccode']
        decimals, marketId =    self.ds.getSecurityInfo (security)
        decimals = int(decimals)
        params = self.ds.getSecurityAlgParams( security )       
        byMarket =              params['entryByMarket']
        entryTimeSeconds =      params['entryTimeSeconds']
        exitTimeSeconds =       params['exitTimeSeconds'] 
        quantity =              params['positionQuantity']
        k =                     params['stopLossCoefficient']
        marginsByHour =         params['positionMargin']
        correctionByHour =      params['correction']
        spreadByHour =          params['spread']
        smallDelta =            params['smallDelta']
        limitToAcceptFallingOfPrice = params['limitToAcceptFallingOfPrice']

        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = dt.datetime.now(moscowTimeZone)
        moscowHour = moscowTime.hour
        
        spread =        spreadByHour[str(moscowHour)]
        correction =    correctionByHour[str(moscowHour)]
        deltaForExit =  marginsByHour[str(moscowHour)]

        exitTime = moscowTime + dt.timedelta(seconds = exitTimeSeconds)
        stoploss = 0.0
        exitPrice =  0.0
        entryPrice =  0.0
        
        status = 0
        predictions = copy.deepcopy(security['predictions'][longestPeriod])    
        byMarket = False            
        fluctuation = predictions[-1]
        numWindowSize = fluctuation['samplingWindow'].shape[0]
        indexPenultimate = numWindowSize - 2
                
        indexLastPeak = fluctuation['peak_idx'][-1] if fluctuation['peak_idx'].any() else 0
        indexLastValley = fluctuation['valley_idx'][-1] if fluctuation['valley_idx'].any() else 0
   
        if indexLastPeak == indexPenultimate and indexLastPeak != indexLastValley :
            status = 1
        elif indexLastValley == indexPenultimate and indexLastPeak != indexLastValley :
             status = -1
        elif indexLastPeak  == indexLastValley: #if in the same point thera peak and valley
             status = 0     
        else:
             status = 0  
             
        status = self.reviewForHigherfrequency( status, security )
             
        logging.info('last change was = ' + str(status) ) 
        
        takePosition = 'no-go'
        
        takePosition = self.takeDecisionPeaksAndValleys(
            security, status, fluctuation, limitToAcceptFallingOfPrice
        )
        entryPrice = self.getEntryPrice(fluctuation, takePosition, smallDelta)
            
        if takePosition == 'long':
            exitPrice = entryPrice  + deltaForExit
            stoploss = entryPrice  - k * deltaForExit                
        elif takePosition == 'short':
            exitPrice = entryPrice  - deltaForExit
            stoploss = entryPrice  + k * deltaForExit                
            
        nogoHours = [7,18,19,21]
        if moscowHour in nogoHours:
            logging.info('we are in a no-go hour ...')  
            takePosition = 'no-go' 
            
        logging.info('exitPrice:  '+str(exitPrice))  
        logging.info('entryPrice: '+str(entryPrice)) 
        position = tp.Position(
            takePosition, board, seccode, marketId, entryTimeSeconds, 
            quantity, entryPrice, exitPrice , stoploss, decimals, exitTime,
            correction, spread, byMarket
        )
        logging.info( 'dolph decides: ' + str(position))    
            
        return position
    
    def takeDecisionPeaksAndValleys(self, security, status,fluctuation, limitToAcceptFallingOfPrice ):
            
        seccode = security['seccode']
        openPosition = self.tp.isPositionOpen( seccode )
        lastClosePrice = self.getLastClosePrice(seccode)
        position = self.tp.getMonitoredPositionBySeccode(seccode)

        takePosition = 'no-go' 
        
        if status == 1:
            if openPosition == True:
                if (security['lastPositionTaken'] ==  'short'):
                    takePosition = 'no-go'
                else:
                    takePosition = 'close-counterPosition'  
                    security['lastPositionTaken'] = None
            else:
                takePosition= 'short' 
                security['lastPositionTaken'] = takePosition

        elif status == -1:
            if openPosition == True:
                if (security['lastPositionTaken'] ==  'long'):
                    takePosition = 'no-go'
                else:
                    takePosition = 'close-counterPosition' 
                    security['lastPositionTaken'] = None
            else:             
                takePosition= 'long' 
                security['lastPositionTaken'] = takePosition
        
        elif status == 0:
            if openPosition == True and position is not None:
                entryPrice = position.entryPrice
                if ( abs( entryPrice - lastClosePrice ) < limitToAcceptFallingOfPrice):
                    takePosition = 'no-go' 
                else:
                    takePosition = 'close-counterPosition'
                    logging.info('entryprice' + str(entryPrice))
                    logging.info('lastClosePrice' + str(lastClosePrice))
                    logging.info('AVALANCHE has happened!')
        else:
            logging.info("this should happen for " + seccode )
            
        return takePosition
  
    
    def getOnlyLastClosePrice (self,fluctuation):
        dataInSamplinWindow = fluctuation['samplingWindow']
        currentClose = dataInSamplinWindow.iloc[-1].EndPrice
        return currentClose
    
    
    def getEntryPrice(self,fluctuation, takePosition,smallDelta ):  
   
        dataInSamplinWindow = fluctuation['samplingWindow']
        currentClose = dataInSamplinWindow.iloc[-1].EndPrice
        currentMin = dataInSamplinWindow.iloc[-1].MinPrice
        currentMax = dataInSamplinWindow.iloc[-1].MaxPrice

        if (takePosition=="long"):
            entryPricePV = currentClose + smallDelta
        elif(takePosition=="short"):
            entryPricePV = currentClose - smallDelta
        else:
            entryPricePV=currentClose 
        return entryPricePV
        
    
    
    def takePosition (self):
        
        for sec in self.securities:            
            position = self.evaluatePosition(sec)            
            action = position.takePosition
            if action not in ['long','short','close','close-counterPosition']:
                logging.info( action + ' position, nothing to do')
                continue            
            if self.MODE == 'OPERATIONAL' :
                logging.info('sending a "' + action +'" to Trading platform ...')
                self.tp.processPosition(position)              
        
           
    
if __name__== "__main__":

    securities = [] 
    securities.append( {'board':'FUT', 'seccode':'SRU1', 'label': 'SBERBANK'} )
    # securities.append( {'board':'FUT', 'seccode':'GZM1'} ) 
    # securities.append( {'board':'FUT', 'seccode':'SiM1'} ) 

    dolph = Dolph( securities )

    while True:
        dolph.dataAcquisition()
        dolph.predict()
        # dolph.displayPredictions()
        dolph.takePosition()         
        
        