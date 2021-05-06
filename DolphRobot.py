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
import TrendViewer as tv
import NeuronalNet as nn
import NeuronalNet_v2 as nn_v2
import NeuronalNet_v3 as nn_v3
import NeuronalNet_v5 as nn_v5
import NeuronalNet_v6 as nn_v6
import NeuronalNet_v9 as nn_v9
import NeuronalNet_v10 as nn_v10
import peaks_and_valleys as fluctuationModel

class Dolph:
    def __init__(self, securities):
        
        self.securities = securities
        
        # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TRAIN_OFFLINE' | 'OPERATIONAL'
        self.MODE = 'OPERATIONAL' 

        self.numTestSample = 1300
        self.since = dt.date(year=2020    ,month=2,day=1)
        self.between_time = ('07:00', '23:50')
        self.TrainingHour = 10
    
        if self.MODE == 'TRAIN_OFFLINE' or self.MODE == 'TEST_OFFLINE':
            
            if self.TrainingHour in range(9,14):
                self.between_time = ('07:00', '14:00')
            else:
                self.between_time = ('14:00', '23:00')
   
               
        self.periods = ['1Min','2Min']

        self.data = {}
        self.inputDataTest = {}
        self.lastUpdate = None
        self.currentTestIndex = 0  
        
        logFormat = '%(asctime)s | %(levelname)s | %(funcName)s |%(message)s'
        logging.basicConfig(
            level = logging.INFO , 
            # level = logging.DEBUG , 
            format = logFormat,
            handlers=[  
                logging.FileHandler("./log/Dolph.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('running on mode: ' + self.MODE)

        self.ds = ds.DataServer()
        self.plotter = tv.TrendViewer(self.periods, self.positionAssestment)
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
        
        if ( alg == 'NeuronalNet' ):
            self.getData = self.ds.resampleDataFrames    
            self.getTrainingModel = nn.trainAndPredict
            self.showPrediction = self.plotter.displayPrediction
        elif (alg == 'NeuronalNet_v2' ):
            self.getData = self.ds.resampleDataFrames    
            self.getTrainingModel = nn_v2.trainAndPredict
            self.showPrediction =  self.plotter.displayPrediction_v2
        elif (alg == 'NeuronalNet_v3' ):
            self.getData = self.ds.resampleDataFrames  
            self.getTrainingModel = nn_v3.NeuronalNet_v3_Model
            self.showPrediction = self.plotter.displayPrediction_v3
        elif (alg == 'NeuronalNet_v5' ):
            self.getData = self.ds.retrieveData  
            self.getTrainingModel= nn_v5.Model
            self.showPrediction = self.plotter.displayPrediction_v5
        elif (alg == 'NeuronalNet_v6' ):
            self.getData = self.ds.searchData
            self.getTrainingModel = nn_v6.MLModel
            self.showPrediction = self.plotter.displayPrediction_v6
        elif (alg == 'NeuronalNet_v9' ):
            self.getData = self.ds.searchData
            self.getTrainingModel = nn_v9.MLModel
            self.showPrediction = self.plotter.displayPrediction_v9
        elif (alg == 'NeuronalNet_v10' ):
            self.getData = self.ds.searchData
            self.getTrainingModel = nn_v10.MLModel
            self.showPrediction = self.plotter.displayPrediction_v10
        elif (alg == 'peaks_and_valleys' ):
            self.getData = self.ds.searchData
            self.getTrainingModel = fluctuationModel.Model
            self.showPrediction = self.plotter.displayNothing    
            
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
        since = dt.date.today()
        periods = self.periods
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
            position.exitPrice =position.entryPrice+deltaForExit
            position.stoploss = position.entryPrice  + k * deltaForExit
            
        
        
        
        logging.info('sending '+position.takePosition+' to Trading platform')
        self.tp.processPosition(position)     
       
    
           
    def isSufficientData (self, dataFrame):
        
        periods = self.periods
        longestPeriod = periods[-1]
        dataFrame = dataFrame[longestPeriod]    
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
        periods = self.periods
        period = periods[-1]        
        numPeriod = int(period[0])
        dataFrame = dfs[period]
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
        
        since = _.since
        periods = _.periods
        securities = _.securities
        dfs = {}
        
        if (_.MODE == 'TRAIN_OFFLINE'):

            dfs = _.getData(securities, periods, since, None, _.between_time)
            for p in periods:
                _.data[p] =  dfs[p]                

        elif (_.MODE == 'TEST_OFFLINE'):
            
            if( _.data == {} ):             
                _.inputDataTest = \
                _.getData(securities, periods, since, None, _.between_time )
   
            for p in periods:
                df = _.inputDataTest[p].copy()
                for s in securities:
                    seccode = s['seccode']
                    indexes = df[ df['Mnemonic'] == seccode ].index
                    indexes = indexes[-_.numTestSample :]
                    df.drop(indexes, inplace = True)                    
                _.data[p] = df
            
            _.numTestSample = _.numTestSample - 1
             
            
        elif ( _.MODE == 'TEST_ONLINE' or _.MODE == 'OPERATIONAL'):   
            
            since = dt.date.today() - dt.timedelta(days=3)            
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
        
        p = self.periods[-1]
        
        for sec in self.securities:
            params = self.ds.getSecurityAlgParams( sec )
            self.loadModel(sec, p, params)            
            logging.debug( 'calling the model ...')
            seccode = sec['seccode']
            dataframe = self.data[p]
            df = dataframe[ dataframe['Mnemonic'] == seccode ]
            pred = sec['models'][p].predict( df, sec )            
            self.storePrediction( sec, pred, p, params)
        
   
    def displayPredictions (self):
        
        logging.info( 'plotting a graph ...') 
        p = self.periods[-1]
        
        for sec in self.securities:
            preds = copy.deepcopy(sec['predictions'][p])
            self.showPrediction( preds , p)    
      
    def positionAssestment (self, security, candlePredList,lastCandle ):        
 
        printPrices = False
        exitPrice = 0.0
        entryPrice = 0.0
        decision = 'no-go'
        currentOpen = lastCandle['currentOpen']
        currentHigh = lastCandle['currentHigh']
        currentLow = lastCandle['currentLow']
        currentClose = lastCandle['currentClose']
       
        self.printPrices = False
       
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = dt.datetime.now(moscowTimeZone)
        moscowHour = moscowTime.hour
        moscowMin= moscowTime.minute

        nogoHours = [16]
        if moscowHour in nogoHours:
            logging.info('we are in a no-go hour ...')  
            return entryPrice, exitPrice, decision, printPrices        
          
        params = self.ds.getSecurityAlgParams( security )
        marginsByHour = params['positionMargin']
        deltaForExit= marginsByHour[str(moscowHour)]
#TODO koschmar....
        firstcandle  = candlePredList[0]
          
       

        def checkCandleBlue(firstcandle):
            blue = False 
            if (firstcandle['Close']>firstcandle['Open']):
                blue= True
            return blue
        def checkCandleBlack(firstcandle):
            black = False 
            if ((firstcandle['Close']<firstcandle['Open'])):
                black= True
            return black

        margin=2
        if moscowHour>9:
            #check the color of the current candle
            if (currentClose>=currentOpen): #if this current blue?
                 CandlesBlackcheck=checkCandleBlack(firstcandle)
                 CandlesBluecheck =checkCandleBlue(firstcandle)
                 if (CandlesBlackcheck == True):
                        logging.info('It seems the market will go down..')  
                        entryPrice=currentClose-margin
                        exitPrice = entryPrice - deltaForExit
                        decision='short'
                        printPrices = True
                 if (CandlesBluecheck == True):
                        logging.info('It seems the market will grow up:')                
                        entryPrice = currentClose+margin
                        exitPrice = entryPrice+deltaForExit
                        decision='long'
                        printPrices = True   
                        
            if (currentClose<currentOpen): #if this current black?
                CandlesBlackcheck=checkCandleBlack(firstcandle)
                CandlesBluecheck =checkCandleBlue(firstcandle)
                if (CandlesBlackcheck == True):
                        logging.info('It seems the market will go down..')  
                        entryPrice=currentClose-margin
                        exitPrice = entryPrice - deltaForExit
                        decision='short'
                        printPrices = True
                if (CandlesBluecheck == True):
                        logging.info('It seems the market will grow up:')                
                        entryPrice = currentClose+margin
                        exitPrice = entryPrice+deltaForExit
                        decision='long'
                        printPrices = True
                        
        else:
            if (moscowHour == 9):
                if (moscowMin > 2 and moscowMin <20):            #first three-four candles we will repeat the the first one until 10:20
                    if (currentClose>currentOpen): #if this current blue?
                        logging.info('It seems the market will grow up:')                
                        entryPrice = currentClose+margin
                        exitPrice = entryPrice+deltaForExit
                        decision='long'
                        printPrices = True
                    if (currentClose<currentOpen): #if this current black?
                        logging.info('It seems the market will go down..')  
                        entryPrice = currentClose+margin
                        exitPrice = entryPrice+deltaForExit
                        decision='long'
                        printPrices = True


                                                                    
        return entryPrice, exitPrice, decision, printPrices
    
    def getPositionAssessmentParams(self,predictions):
        
        myCols = ['predictions','time','Mnemonic','EndPrice' ]
        df_in = pd.DataFrame(columns=myCols)
        
        for p in predictions:
            row = pd.DataFrame({
                'time':         p.training_set.original_df['CalcDateTime'],
                'Mnemonic':     p.training_set.original_df['Mnemonic'],
                'StartPrice':   p.training_set.original_df['StartPrice'],
                'EndPrice':     p.training_set.original_df['EndPrice'],
                'MinPrice':     p.training_set.original_df['MinPrice'],
                'MaxPrice':     p.training_set.original_df['MaxPrice'] ,
                  
                'close_t+1':            p.predictions[0][0] #,  
                

                }
            )
            df_in = df_in.append(row)
        df_in['timeDate'] = df_in['time']  
        df_in = df_in.set_index('time')
        df = df_in        
        times = sorted(list(df.index.unique()))
        lastTime = times[-1]    
        
        t1 = lastTime + dt.timedelta(minutes = 1)
        t2 = lastTime + dt.timedelta(minutes = 2)
        t3 = lastTime + dt.timedelta(minutes = 3)
        t4 = lastTime + dt.timedelta(minutes = 4)

        currentOpen = df.iloc[-1].StartPrice
        currentHigh = df.iloc[-1].MaxPrice
        currentLow = df.iloc[-1].MinPrice
        currentClose = df.iloc[-1].EndPrice

        p = df.loc[lastTime]
        candlePredList = [
            # {'Date': t1, 'Open': currentClose, 'High': currentHigh + p['high_t+1'], 'Low': currentLow + p['low_t+1'], 'Close': currentClose + p['close_t+1']}   #,
            {'Date': t1, 'Open': currentClose, 'High': currentClose  + 1, 'Low': currentClose  - 1, 'Close': currentClose + p['close_t+1']}   #,
            # {'Date': t2, 'Open': currentClose + p['close_t+1'], 'High': currentHigh + p['high_t+2'], 'Low': currentLow + p['low_t+2'], 'Close': currentClose + p['close_t+1'] + p['close_t+2']},
            # {'Date': t3, 'Open': currentClose + p['close_t+1'] + p['close_t+2'], 'High': currentHigh + p['high_t+3'], 'Low': currentLow + p['low_t+3'], 'Close': currentClose + p['close_t+1'] + p['close_t+2'] + p['close_t+3']},
            # {'Date': t4, 'Open': currentClose + p['close_t+1'] + p['close_t+2'] + p['close_t+3'], 'High': currentHigh + p['high_t+4'], 'Low': currentLow +p['low_t+4'], 'Close':  currentClose + p['close_t+1'] + p['close_t+2'] + p['close_t+3'] + p['close_t+4']}
        ] 
        
        lastCandle = { 
            'currentOpen' : currentOpen,
            'currentHigh' : currentHigh,
            'currentLow' : currentLow,
            'currentClose' : currentClose                      
        }
        
        return candlePredList,lastCandle        


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
        
        predictions = copy.deepcopy(security['predictions'][longestPeriod])
        model = security['models'][longestPeriod]

        if hasattr(model, 'id') and model.id == 'peaks_and_valleys':
            byMarket = False            
            fluctuation = predictions[-1]
            numWindowSize = fluctuation['samplingWindow'].shape[0]
            indexLastPeak = fluctuation['peak_idx'][-1]
            indexLastValley = fluctuation['valley_idx'][-1]
        
            if indexLastPeak == (numWindowSize - 2) and indexLastPeak  != indexLastValley :
                status = 1
            elif indexLastValley == (numWindowSize - 2) and indexLastPeak  != indexLastValley :
                 status = -1
            elif indexLastPeak  == indexLastValley: #if in the same point thera peak and valley
                 status = 0  
            else:
                 status = 0  
                 
            logging.info('last change was = ' + str(status) ) 
           
            takePosition = self.takeDecisionPeaksAndValleys(security, status, fluctuation,limitToAcceptFallingOfPrice )
            entryPrice = self.getEntryPrice(fluctuation, takePosition, smallDelta)
            if (takePosition == 'short' or takePosition == 'long'):
                security['savedEntryPrice'] = entryPrice
            
                
            
            nogoHours = [16]
            if moscowHour in nogoHours:
                logging.info('we are in a no-go hour ...')  
                takePosition = 'no-go' 
                entryPrice = 0.0

        else:
            candlePredList,lastCandle = self.getPositionAssessmentParams(predictions)
            entryPrice = lastCandle['currentClose']            
            entryPrice, exitPrice, takePosition, printPrices = \
                self.positionAssestment(security,candlePredList,lastCandle)

        if takePosition == 'long':
            exitPrice = entryPrice  + deltaForExit
            stoploss = entryPrice  - k * deltaForExit
            
        elif takePosition == 'short':
            exitPrice = entryPrice  - deltaForExit
            stoploss = entryPrice  + k * deltaForExit
            
        else:
            exitPrice = entryPrice
            stoploss = entryPrice
        logging.info('exitPrice'+str(exitPrice))  
        logging.info('entryPrice'+str(entryPrice)) 
        position = tp.Position(
            takePosition, board, seccode, marketId, entryTimeSeconds, 
            quantity, entryPrice, exitPrice , stoploss, decimals, exitTime,
            correction, spread, byMarket
        )
        logging.info( 'dolph decides: ' + str(position))    
            
        return position
    
    def takeDecisionPeaksAndValleys(self, security, status,fluctuation, limitToAcceptFallingOfPrice ):
        
        distanceBetweenPeekAndValley=2

        openPosition = self.tp.isPositionOpen( security['seccode'] )
        
        if (fluctuation['peak_idx'].size  >= 2):    
            indexLastPeak = fluctuation['peak_idx'][-1]
            index2LastPeak = fluctuation['peak_idx'][-2]
        elif(fluctuation['peak_idx'].size  == 1 ):
            indexLastPeak = fluctuation['peak_idx'][-1]
            index2LastPeak = 0
        else:
            indexLastPeak=0
            index2LastPeak=0
        if (fluctuation['valley_idx'].size  >= 2):   
            indexLastValley = fluctuation['valley_idx'][-1]
            index2LastValley = fluctuation['valley_idx'][-2]
        elif(fluctuation['valley_idx'].size  == 1 ):
            indexLastValley = fluctuation['valley_idx'][-1]
            index2LastValley = 0
        else:
            indexLastValley=0
            index2LastValley=0
            
        takePosition = 'no-go'
        
        if status == 1:
            if openPosition == True:
                if  (security['lastPositionTaken']  =='short' ):
                    takePosition = 'no-go'
                elif (abs(indexLastPeak-indexLastValley) <= distanceBetweenPeekAndValley):
                    takePosition = 'no-go' 
                else:
                    takePosition = 'close'  
                    security['lastPositionTaken'] = None
                    security['savedEntryPrice'] = 0
            else:
                if (abs(indexLastPeak-indexLastValley) <= distanceBetweenPeekAndValley):
                    takePosition = 'no-go'
                else:
                    takePosition= 'short' 
                    security['lastPositionTaken'] = takePosition
                    security['savedEntryPrice'] = 0

        elif (status == -1):
            if (openPosition == True ):
                if (security['lastPositionTaken'] =='long' ):
                    takePosition = 'no-go'
                elif (abs(indexLastValley-indexLastPeak) <= distanceBetweenPeekAndValley):
                    takePosition = 'no-go'
                else: 
                    takePosition = 'close' 
                    security['lastPositionTaken'] = None
                    security['savedEntryPrice'] = 0

            else:
                if (abs(indexLastPeak-indexLastValley) <= distanceBetweenPeekAndValley):
                    takePosition = 'no-go'
                else:
                    takePosition= 'long' 
                    security['lastPositionTaken'] = takePosition
                    security['savedEntryPrice'] = 0
        else:
            if openPosition == True: 
                lastClosePrice = self.getOnlyLastClosePrice(fluctuation)
                if (abs(security['savedEntryPrice'] - lastClosePrice) < limitToAcceptFallingOfPrice):
                    takePosition = 'no-go' 
                else:
                    takePosition = 'close-counterPosition'
                    logging.info('entryprice' + str(security['savedEntryPrice']))
                    logging.info('lastClosePrice' + str(lastClosePrice))
        return takePosition
    
    
    
    def getOnlyLastClosePrice (self,fluctuation):
        dataInSamplinWindow = fluctuation['samplingWindow']
        currentClose = dataInSamplinWindow.iloc[-1].EndPrice
        return currentClose
    
    
    def getEntryPrice(self,fluctuation, takePosition,smallDelta ):  
   
        dataInSamplinWindow = fluctuation['samplingWindow']
        currentClose = dataInSamplinWindow.iloc[-1].EndPrice

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
    # securities.append( {'board':'FUT', 'seccode':'SRM1'} )
    securities.append( {'board':'FUT', 'seccode':'GZM1'} ) 
    # securities.append( {'board':'FUT', 'seccode':'SiM1'} ) 

    dolph = Dolph( securities )

    while True:
        dolph.dataAcquisition()
        dolph.predict()
        dolph.displayPredictions()
        dolph.takePosition()         
        
        