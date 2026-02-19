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
        self.logger = None      
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
        self.open_ai_key=cm.openaikey


    def _init_securities(self):
        
        for sec in self.securities: 
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
            format = '%(asctime)s | %(levelname)s | %(name)s.%(funcName)s |%(message)s',
            handlers=[  
                logging.FileHandler("./log/Dolph.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('running on mode: ' + self.MODE)
        self.logger = logging.getLogger("Dolph")


    def _init_signaling(self):
        
        def signalHandler( signum, frame):
            self.tp.disconnect()
            time.sleep(2.5) 
            logging.info('hasta la vista!')
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
            self.logger.error('Security not found... ' + seccode);
            sys.exit(0)
        
        return sec
        
    
    def getLastClosePrice(self, seccode):

        dataFrame_1min = self.data['1Min']  # Get the 1-minute data
        
        # Filter the data for the specified security code
        df_1min = dataFrame_1min[dataFrame_1min['mnemonic'] == seccode]
        
        # Check if df_1min is empty
        if df_1min.empty:
            self.logger.error(f"No data found for seccode: {seccode}")
            return None  # Or handle this case differently depending on your requirements
    
        # Get the last close price
        timelast1Min = df_1min.index[-1]
        timelast1Min = timelast1Min.to_pydatetime()
        LastClosePrice = df_1min['endprice'].iloc[-1]
        self.logger.debug(f'seccode={seccode} last sample at UTC-time={timelast1Min}, Close={LastClosePrice}')
        
        # Check if the LastClosePrice is None
        if pd.isnull(LastClosePrice):
            self.logger.error(f'LastClosePrice is None for seccode: {seccode}')
            sys.exit(0)
        
        return LastClosePrice


    def getLastClose(self, seccode):
        
        dataFrame_1min = self.data['1Min']
        
        # Filter the data for the specified security code
        df_1min = dataFrame_1min[dataFrame_1min['mnemonic'] == seccode]
        
        # Check if df_1min is empty
        if df_1min.empty:
            self.logger.error(f"No data found for seccode: {seccode}")
            return None  # Or handle this case differently depending on your requirements
    
        # Get the last close price
        timelast1Min = df_1min.index[-1]
        timelast1Min = timelast1Min.to_pydatetime()
        LastClosePrice = df_1min['endprice'].iloc[-1]
 
        
        return timelast1Min , LastClosePrice
        
    
    def onCounterPosition(self, position2invert ):
        
        position = copy.deepcopy(position2invert)
        if (position2invert.takePosition == "long"):           
            position.takePosition = "short"
        elif (position2invert.takePosition  == "short"):
            position.takePosition = "long"
        else:
            self.logger.error( "takePosition must be either long or short")
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
            self.logger.info('this shouldnt happen ' + position.takePosition )
        
        self.logger.info('sending '+position.takePosition+' to Trading platform')
        self.tp.processPosition(position)     
       

    def dataAcquisition(self):        
        
        self.ds.syncData( self.data )


    def _getPredictionModel(self, sec, period):
        
        sec['params']['period'] = period 
        sec['models'][period] = pm.initPredictionModel( self.data, sec, self)
        msg = f"loading training & prediction params: {sec['params']}"
        self.logger.info( msg )        

    
    def storePrediction(self, sec, prediction, period):

        sec['predictions'][period].append(prediction)
   
    
    def loadModel(self, sec, period):        
   
        if period not in sec['models']:
            self.logger.info('model loaded for the first time') 
            self._getPredictionModel(sec, period )            


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
    
    # def isBetterToClosePosition(self, security):
        
    #     seccode = security['seccode']
    #     params = security['params']

    #     position = self.tp.getMonitoredPositionBySeccode(seccode)
    #     if position is None :
    #         self.logger.error(f'seccode={seccode} has an open-position, but there is no MonitoredPosition')
    #         return True
        
    #     lastClosePrice = self.getLastClosePrice(seccode)
    #     entryPrice = position.entryPrice
    #     factorMargin_Position = params['positionMargin']
    #     stopLossCoefficient = params['stopLossCoefficient']
    #     limitToAcceptFallingOfPrice = entryPrice * factorMargin_Position * stopLossCoefficient
    #     decision = False
    #     if ( abs( entryPrice - lastClosePrice ) > limitToAcceptFallingOfPrice):
    #         decision = True
    #         self.logger.info(f'entryPrice: {entryPrice},lastClosePrice: {lastClosePrice}')
        
    #     return decision
    
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
        #isBetterToClose = self.isBetterToClosePosition(security) if openPosition else False
                
        if openPosition and security['lastPositionTaken'] == prediction :

            takePosition = 'no-go'
                     
        # elif openPosition and isBetterToClose:            
        #     takePosition = 'close'
        #     security['lastPositionTaken'] = takePosition
        
        elif not openPosition:
            
            takePosition = prediction
            security['lastPositionTaken'] = takePosition
       
        self.logger.debug(f'{takePosition}')
        return takePosition
  
    
    def positionExceedsBalance (self, position):

        exceeds = True if position.quantity == 0 else False

        net_balance = self.tp.get_net_balance()
        if net_balance == 0 : return True

        side = position.takePosition
        same_side_exposure = sum(
            p.quantity * p.entryPrice for p in self.tp.monitoredPositions
            if p.takePosition == side
        )
        new_exposure = position.quantity * position.entryPrice
        total_side_exposure = same_side_exposure + new_exposure

        if total_side_exposure > net_balance :
            exceeds = True
            self.logger.error(f"{side} exposure {total_side_exposure} (existing={same_side_exposure} + new={new_exposure}) > net_balance {net_balance}")

        return exceeds
    
    
    def positionAssessment (self, security):
        
        seccode = security['seccode']
        params = security['params']

        timeClose, priceClose = self.getLastClose( seccode)
        cash_balance = self.tp.get_cash_balance()
        net_balance = self.tp.get_net_balance()
        factorMargin_Position = params['positionMargin']        
        
        cash_4_position = net_balance * cm.factorPosition_Balance        
        quantity = round(cash_4_position / priceClose)
        margin = priceClose * factorMargin_Position

        printMargin = "{0:0.{prec}f}".format(factorMargin_Position, prec=5)

        m = f"cash_balance={cash_balance} net_balance={net_balance} quantity={quantity} " 
        m += f"factor-margin={printMargin} UTC-Time={timeClose} priceClose={priceClose}" 

        if cash_balance == 0 or net_balance == 0: 
            self.logger.warning(f"seccode={seccode} condition=(cash_balance == 0 or net_balance == 0) {m}")
            return 0 , 0 
        
        new_position_value = quantity * priceClose
        long_exposure = sum(p.quantity * p.entryPrice for p in self.tp.monitoredPositions if p.takePosition == 'long')
        short_exposure = sum(p.quantity * p.entryPrice for p in self.tp.monitoredPositions if p.takePosition == 'short')
        long_would_exceed = (long_exposure + new_position_value) > net_balance
        short_would_exceed = (short_exposure + new_position_value) > net_balance
        if long_would_exceed and short_would_exceed:
            self.logger.warning(f"seccode={seccode} both sides full: long_exposure={long_exposure} short_exposure={short_exposure} new={new_position_value} net_balance={net_balance} {m}")
            return 0 , 0

        self.logger.info(f"seccode:{seccode} {m}")

        return quantity, margin
    
    def get_evaluation_parameters(self, security):
        
        try:
            longestPeriod = self.periods[-1]
            board, seccode, params = security['board'], security['seccode'], security['params']
            quantity, margin = self.positionAssessment(security)
            # FIXME: Can the decimals be automatically calculated?            
            decimals, marketId = security['decimals'], security['market']
            # FIXME: Are these parameters automatically calculated?
            exitTimeSeconds = params.get('exitTimeSeconds', cm.exitTimeSeconds)
            k = params.get('stopLossCoefficient', cm.stopLossCoefficient ) 
            correction = params.get('correction', cm.correction) 
            spread = params.get('spread', cm.spread)
            
            decimals = int(decimals)                   
            ct = self.tp.getTradingPlatformTime()
            exitTime = ct + dt.timedelta(seconds=exitTimeSeconds)
            
            return (longestPeriod, board, seccode, exitTimeSeconds, 
                quantity, k, decimals, marketId, spread, correction, margin, exitTime )
        
        except Exception as e:
            self.logger.error("Failed to get_evaluation_parameters: %s", e)
            k = margin = quantity = correction = spread = decimals = marketId = ct = exitTime = 0            
            
            return (longestPeriod, board, seccode, exitTimeSeconds, 
                quantity, k, decimals, marketId, spread, correction, margin, exitTime )
       



    def evaluatePosition (self, security):        

        (longestPeriod, board, seccode, exitTimeSeconds, 
         quantity, k, decimals, marketId, spread, correction, margin, exitTime ) = self.get_evaluation_parameters(security)
            
        prediction = copy.deepcopy(security['predictions'][longestPeriod])
        takePosition = self.takeDecision( security, prediction)
        entryPrice = self.getEntryPrice(seccode, takePosition )
        client = self.tp.getClientId()
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
            takePosition, board, seccode, marketId,
            quantity, entryPrice, exitPrice, stoploss, decimals, client, 
            exitTime, correction, spread, byMarket 
        )
        
        if takePosition in ['long','short']:            
            if self.positionExceedsBalance(position):
                position.takePosition = 'no-go' 
        
        self.logger.debug(f'decision: {position}')    
            
        return position
    
    
    def takePosition (self):
        
        for sec in self.securities:            
            
            position = self.evaluatePosition(sec)            
            action = position.takePosition            
            if action not in ['long','short','close','close-counterPosition']:
                self.logger.info(f"seccode:{position.seccode} action={action}, nothing to do ...")
                continue            

            self.logger.info(f'seccode:{position.seccode} sending a {position} to the Trading platform ...')
            self.tp.processPosition(position)  
       
        
    def initDB (self):

        self.logger.info('initDB, getting Securities ids...')

        for sec in self.securities:
            board, seccode = sec['board'], sec['seccode']
            sec['id'] = self.ds.getSecurityIdSQL(board, seccode)        
        
        
        if self.MODE != 'INIT_DB' : 
            return
        
        self.logger.info('init database according to Trading platform ...')
          
        for sec in self.securities:
            
            self.logger.info(f"getting candles for {sec['seccode']} ... ")
            candles = self.tp.get_candles(sec, self.since, self.until, period = '1Min')
            self.logger.info(f"storing candles for {sec['seccode']} ... ")
            self.ds.store_candles(candles,sec) 
    
        sys.exit(0)           
        raise SystemExit("Stopping the program") 


    def setSecurityParams(self, seccode, **params):
        """ 
        How to use:          
        params = {'positionMargin': 0.01}
        self.dolph.setSecurityParams('INTC', **params )        
        params = {'stopLossCoefficient': 3 }
        self.dolph.setSecurityParams('INTC', **params )        
        """
        for sec in self.securities:
            if sec['seccode'] == seccode:
                for key, value in params.items():
                    if key in sec['params']:
                        m = f"Updating {seccode}: {sec['params'][key]} -> {value}"
                        logging.info(m)
                        sec['params'][key] = value
                    else:
                        logging.warning(f"Key '{key}' does not exist in {seccode}")
                return
        logging.error(f"Security with seccode '{seccode}' not found")
    

def main():

    dolph = Dolph()  

    iteration = 0

    while True:
        iteration += 1
        
        try:
            dolph.logger.info(f"{'='*60}")
            dolph.logger.info(f"MAIN LOOP ITERATION {iteration} - START")
            dolph.logger.info(f"{'='*60}")
            
            dolph.logger.info(f"[Iter {iteration}] Step 1/3: Data Acquisition")
            dolph.dataAcquisition()
            dolph.logger.info(f"[Iter {iteration}] Step 1/3: ✓ COMPLETED")
            
            dolph.logger.info(f"[Iter {iteration}] Step 2/3: Predict")
            dolph.predict()
            dolph.logger.info(f"[Iter {iteration}] Step 2/3: ✓ COMPLETED")
                        
            dolph.logger.info(f"[Iter {iteration}] Step 3/3: Take Position")
            dolph.takePosition()
            dolph.logger.info(f"[Iter {iteration}] Step 3/3: ✓ COMPLETED")
            
            dolph.logger.info(f"{'='*60}")
            dolph.logger.info(f"MAIN LOOP ITERATION {iteration} - FINISHED")
            dolph.logger.info(f"{'='*60}\n")
            
        except KeyboardInterrupt:
            dolph.logger.info("Keyboard interrupt received, shutting down...")
            break
            
        except Exception as e:
            dolph.logger.error(f"\n{'!'*60}")
            dolph.logger.error(f"ERROR IN MAIN LOOP ITERATION {iteration}")
            dolph.logger.error(f"{'!'*60}")
            dolph.logger.error(f"Error: {e}")
            import traceback
            dolph.logger.error(traceback.format_exc())
            dolph.logger.error(f"{'!'*60}\n")
            
            dolph.logger.info("Waiting 10 seconds before retrying...")
            time.sleep(10)

   
if __name__ == "__main__":
    main()
