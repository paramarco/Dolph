# -*- coding: utf-8 -*- 
# @author: Dolph investments

import logging
import sys
import os
import signal
import time
import gc; gc.collect()
import datetime as dt
import copy
import pandas as pd
import DataManagement.DataServer as ds
import Configuration.Conf as cm
import TradingPlatforms.TradingPlatform as tp 
import PredictionModels.PredictionModel as pm
import DataVisualization.TrendViewer as tv

class Dolph:

    def __init__(self):
        
        self._init_configuration()        
        self._init_logging()
        self.ds = ds.DataServer()
        self.tp = tp.initTradingPlatform( self.onCounterPosition )   
        self.initDB()
        self._init_securities() 
        self.tv = tv.TrendViewer( self.evaluatePosition )
        self.data = {}
        self._init_signaling()

        

    def _init_configuration(self):
        
        self.MODE = cm.MODE
        self.numTestSample = cm.numTestSample
        self.since = cm.since
        self.until = cm.until
        self.between_time = cm.between_time
        self.TrainingHour = cm.TrainingHour
        self.periods = cm.periods
        self.securities = cm.securities
        self.currentTestIndex = cm.currentTestIndex
    

    def _init_securities(self):
        
        for sec in self.securities: 
            sec['params'] = self.ds.getSecurityAlgParams( sec )
            sec['models'] = {}
            sec['predictions'] = {}
            seccode = sec['seccode']
            positions = self.tp.get_PositionsByCode(seccode)
            if not positions:
                sec['lastPositionTaken'] = None
            else:
              for p in positions :
                  sec['lastPositionTaken'] = p.takePosition
            
            for p in self.periods:
                sec['predictions'][p] = []   
            
            logging.info(str(sec))

    
    def _init_logging(self):
        
        logging.basicConfig(            
            level = cm.logLevel ,
            format = '%(asctime)s | %(levelname)s | %(funcName)s |%(message)s',
            handlers=[  
                logging.FileHandler("./log/Dolph.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('running on mode: ' + self.MODE)
        

    def _init_signaling(self):
        
        def signalHandler( signum, frame):
            self.tp.disconnect()
            time.sleep(2.5) 
            print ('hasta la vista!')
            sys.exit()
        
        signal.signal(signal.SIGINT, signalHandler)     
        

    # return next((s for s in self.securities if s['seccode'] == seccode), None)
    def getSecurityBySeccode(self, seccode):
        sec = None
        for s in self.securities: 
            if seccode == s['seccode']:
                sec = s
                break
        
        if sec is None: 
            logging.error('Security not found... ' + seccode);
            sys.exit(0)
        
        return sec
        
    
    def getLastClosePrice(self, seccode):
        # Get the 1-minute data
        dataFrame_1min = self.data['1Min']
        #logging.debug(dataFrame_1min.head)
        
        # Filter the data for the specified security code
        df_1min = dataFrame_1min[dataFrame_1min['mnemonic'] == seccode]
        #logging.debug(df_1min.head)
        
        # Check if df_1min is empty
        if df_1min.empty:
            logging.error(f"No data found for seccode: {seccode}")
            return None  # Or handle this case differently depending on your requirements
    
        # Get the last close price
        timelast1Min = df_1min.index[-1]
        timelast1Min = timelast1Min.to_pydatetime()
        LastClosePrice = df_1min['endprice'].iloc[-1]
        logging.debug(f'time last {timelast1Min}, Close: {LastClosePrice}')
        
        # Check if the LastClosePrice is None
        if pd.isnull(LastClosePrice):
            logging.error(f'LastClosePrice is None for seccode: {seccode}')
            sys.exit(0)
        
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
        lastClosePrice = self.getLastClosePrice(position.seccode)                
        params = sec['params']
        ct = self.tp.getTradingPlatformTime()        
        deltaForExit =  params['positionMargin'][str(ct.hour)]
        position.exitTime = ct + dt.timedelta(seconds = params['exitTimeSeconds'])
        smallDeltaExtreamCase = params['smallDeltaExtreamCase']
        
        if position.takePosition == 'long':
            position.entryPrice = lastClosePrice + smallDeltaExtreamCase
            position.exitPrice = position.entryPrice + deltaForExit
            position.stoploss = position.entryPrice - params['stopLossCoefficient'] * deltaForExit
            
        elif position.takePosition == 'short':
            position.entryPrice = lastClosePrice - smallDeltaExtreamCase
            position.exitPrice =position.entryPrice-deltaForExit
            position.stoploss = position.entryPrice + params['stopLossCoefficient'] * deltaForExit
        else:
            logging.info('this shouldnt happen ' + position.takePosition )
        
        logging.info('sending '+position.takePosition+' to Trading platform')
        self.tp.processPosition(position)     
       

    def dataAcquisition(self):        
        
        self.ds.syncData( self.data )


    def _getPredictionModel(self, sec, period):
        
        sec['params']['period'] = period 
        sec['models'][period] = pm.initPredictionModel( self.data, sec, self)
        msg = f"loading training & prediction params: {sec['params']}"
        logging.debug( msg )        

    
    def storePrediction(self, sec, prediction, period):

        sec['predictions'][period].append(prediction)
   
    
    def loadModel(self, sec, period):        
   
        if period not in sec['models']:
            self._getPredictionModel(sec, period )
        else:
            logging.debug('model already loaded') 


    def predict( self ):
        
        for period in self.periods:
            for sec in self.securities:
                self.loadModel(sec, period)            
                prediction = sec['models'][period].predict( self.data[period], sec, period )            
                self.storePrediction( sec, prediction, period)
        
   
    def displayPredictions (self):
        
        period = self.periods[-1]
        
        for security in self.securities:
            prediction = copy.deepcopy(security['predictions'][period])
            self.tv.showPrediction( prediction , period, security)    
  
    
    def getEntryPrice(self, seccode, takePosition ):    
        
        currentClose = self.getLastClosePrice( seccode)
        if currentClose is None:
            return None
        smallDelta = currentClose * 0.001
    
        if (takePosition=="long"):
            entryPricePV = currentClose + smallDelta
        elif(takePosition=="short"):
            entryPricePV = currentClose - smallDelta
        else:
            entryPricePV=currentClose 
        return entryPricePV
    
    def isBetterToClosePosition(self, security):
        
        seccode = security['seccode']
        position = self.tp.getMonitoredPositionBySeccode(seccode)
        if position is None :
            return True
        
        lastClosePrice = self.getLastClosePrice(seccode)
        entryPrice = position.entryPrice
        limitToAcceptFallingOfPrice = entryPrice * 0.01  # 1% limit
        decision = False
        if ( abs( entryPrice - lastClosePrice ) > limitToAcceptFallingOfPrice):
            decision = True
            logging.info(f'entryPrice: {entryPrice},lastClosePrice: {lastClosePrice}')
        
        return decision
    
    def isPredictionInOppositeDirection(self, prediction, security ) :
        
        inOppositeDirection = False
        
        if security['lastPositionTaken'] == 'long' and prediction == 'short':
            inOppositeDirection = True
        
        if security['lastPositionTaken'] == 'short' and prediction == 'long':
            inOppositeDirection = True
        
        return inOppositeDirection
    
  
    def takeDecision(self, security, prediction ):
            
        seccode = security['seccode']
        openPosition = self.tp.isPositionOpen( seccode )
        takePosition = 'no-go' 
        prediction = prediction[-1]
        isBetterToClose = self.isBetterToClosePosition(security) if openPosition else False
        isPredictionSwap = self.isPredictionInOppositeDirection(prediction, security)
                
        if openPosition and security['lastPositionTaken'] == prediction :

            takePosition = 'no-go'
                     
        elif openPosition and isPredictionSwap and isBetterToClose:
            
            takePosition = 'close'
            security['lastPositionTaken'] = takePosition
        
        elif not openPosition:
            
            takePosition = prediction
            security['lastPositionTaken'] = takePosition

        else:
            logging.error("this should not happen for " + seccode )
                    
        logging.info(f'{takePosition}')
        return takePosition
  
    
    def positionExceedsBalance (self, position):
        
        exceeds = True if position.quantity == 0 else False
        
        cash_balance = self.tp.get_cash_balance()
        positions = self.tp.get_PositionsByCode(position.seccode)
        
        cash_positions = position.quantity * position.entryPrice
        for p in positions :
            cash_positions += p.quantity * p.entryPrice
            
        if cash_positions > cash_balance : exceeds = True 
        
        return exceeds 
    
    
    def calculateMarginQuatityOfPosition (self, security):
        
        seccode = security['seccode']
        currentClose = self.getLastClosePrice( seccode)
        cash_balance = self.tp.get_cash_balance()
        cash_4_position = cash_balance * cm.factorPosition_Balance
        quantity = round(cash_4_position / currentClose)
        margin = currentClose * cm.factorMargin_Position

        return quantity, margin
    
    def get_evaluation_parameters(self, security):
        
        try:
            longestPeriod = self.periods[-1]
            board, seccode, params = security['board'], security['seccode'], security['params']
            quantity, margin = self.calculateMarginQuatityOfPosition(security)
            # FIXME: Can the decimals be automatically calculated?
            decimals, marketId = self.ds.getSecurityInfo(security) 
            # FIXME: Are these parameters automatically calculated?
            entryTimeSeconds = params.get('entryTimeSeconds', cm.entryTimeSeconds)
            exitTimeSeconds = params.get('exitTimeSeconds', cm.exitTimeSeconds)
            margin = params.get('positionMargin', margin)
            k = params.get('stopLossCoefficient', cm.stopLossCoefficient ) 
            correction = params.get('correction', cm.correction) 
            spread = params.get('spread', cm.spread)
            
            decimals = int(decimals)                   
            ct = self.tp.getTradingPlatformTime()
            exitTime = ct + dt.timedelta(seconds=exitTimeSeconds)
        
        except Exception as e:
            k = margin = quantity = correction = spread = decimals = marketId = ct = exitTime = 0
            logging.error("Failed to get_evaluation_parameters: %s", e)
       
        return (longestPeriod, board, seccode, entryTimeSeconds, exitTimeSeconds, 
                quantity, k, decimals, marketId, spread, correction, margin, exitTime )


    def evaluatePosition (self, security):        

        (longestPeriod, board, seccode, entryTimeSeconds, exitTimeSeconds, 
         quantity, k, decimals, marketId, spread, correction, margin, exitTime ) = self.get_evaluation_parameters(security)
            
        prediction = copy.deepcopy(security['predictions'][longestPeriod])
        takePosition = self.takeDecision( security, prediction)
        entryPrice = self.getEntryPrice(seccode, takePosition )
        byMarket = False
        stoploss = entryPrice
        exitPrice = entryPrice
        
        if entryPrice is None:
            takePosition == 'no-go'
            
        elif takePosition == 'long':
            exitPrice = entryPrice  + margin
            stoploss = entryPrice  - k * margin                
            
        elif takePosition == 'short':
            exitPrice = entryPrice  - margin
            stoploss = entryPrice  + k * margin                
       
        position = tp.Position(
            takePosition, board, seccode, marketId, entryTimeSeconds, 
            quantity, entryPrice, exitPrice, stoploss, decimals, exitTime,
            correction, spread, byMarket 
        )
        
        if self.positionExceedsBalance(position):
            position.takePosition = 'no-go'
        
        logging.info( 'dolph decides: ' + str(position))    
            
        return position
    
    
    def takePosition (self):
        
        for sec in self.securities:            
            
            position = self.evaluatePosition(sec)            
            action = position.takePosition            
            if action not in ['long','short','close','close-counterPosition']:
                logging.info( action + ' position, nothing to do')
                continue            

            logging.info('sending a "' + action +'" to Trading platform ...')
            self.tp.processPosition(position)  
       
        
    def initDB (self):
        
        if self.MODE != 'TEST_ONLINE' : 
            return
        
        logging.info('init database according to Trading platform ...')

        filePath = "./TradingPlatforms/Alpaca/AlpacaTickers.json"
        self.ds.insert_alpaca_tickers(filePath)             
          
        for sec in self.securities :
            
            logging.debug("getting candles ... ")
            candles = self.tp.get_candles(sec, self.since, self.until, period = '1Min')
            logging.debug("storing candles ... ")
            self.ds.store_candles(candles,sec) 
    
        sys.exit(0)           
        raise SystemExit("Stopping the program") 

        
    
if __name__== "__main__":

    dolph = Dolph()

    while True:
        
        dolph.dataAcquisition()
        dolph.predict()
        dolph.displayPredictions()
        dolph.takePosition()    
    
    