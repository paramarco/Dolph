# -*- coding: utf-8 -*- 
# @author: Dolph investments

import logging
import sys
import signal
import gc; gc.collect()
import datetime
import copy

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


class Dolph:
    def __init__(self, securities):
    
        # MODE := 'TEST_ONLINE' | TEST_OFFLINE' | 'TEST_ONLINE' | 'OPERATIONAL'

        self.MODE = 'OPERATIONAL' 

        self.numTestSample = 500
        self.since = datetime.date(year=2020,month=3,day=1)
        self.between_time = ('07:30', '23:00')


        # self.periods = ['1Min','2Min','3Min']
        self.periods = ['1Min']

        self.data = {}
        self.inputDataTest = {}
        self.lastMinuteUpdate = None
        self.currentTestIndex = 0  

        self.predictions = {}
        for p in self.periods:
            self.predictions[p] = []
        
        self.df_prediction = None
        self.models = {}
        
        logFormat = '%(asctime)s | %(levelname)s | %(funcName)s |%(message)s'
        logging.basicConfig(
            # level = logging.INFO , 
            level = logging.DEBUG , 
            format = logFormat,
            handlers=[  
                logging.FileHandler("./log/Dolph.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info('running on mode: ' + self.MODE)
        
        self.ds = ds.DataServer()
        self.plotter = tv.TrendViewer(self.periods)
        self.securities = securities
        self.getData = None
        self.getTrainingModel = None
        self.showPrediction = None      
               
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
        else:
            raise RuntimeError('algorithm not found')

        connectOnInit = False
        if (self.MODE == 'TEST_ONLINE' or self.MODE == 'OPERATIONAL' ):
            connectOnInit = True
            
        self.tp = tp.TradingPlatform(
            self.target, 
            self.securities, 
            self.onHistoryCandleRes,
            connectOnInit
        )
        
        def signalHandler(signum, frame):
            self.tp.disconnect()
            print ('hasta la vista!')
            sys.exit()
        
        signal.signal(signal.SIGINT, signalHandler)
        


    def onHistoryCandleRes(self, obj):
        logging.debug( repr(obj) )            
        self.ds.storeCandles(obj)
           
    def isSufficientData (self, dataFrame):
    
        isNew = False
        msg = 'there is only %s samples now, you need at least %s samples for '
        msg+= 'the model to be able to predict'

        numSamplesNow = len(dataFrame.index)
        if ( numSamplesNow < self.minNumPastSamples ):
            logging.warning(msg, numSamplesNow,self.minNumPastSamples)            
            return False
        
        currentDateDataframe = dataFrame.tail(1).index
        currentDate = currentDateDataframe.to_pydatetime()
        currentMinute = currentDate[0].strftime("%M") 
        logging.debug(' current: '  + str( currentMinute) )
        logging.debug(' last: '     + str( self.lastMinuteUpdate) )
        if ( currentMinute != self.lastMinuteUpdate):
            self.lastMinuteUpdate = currentMinute
            isNew = True
        
        return isNew    

    def dataAcquisition(_):
        
        num = _.numTestSample
        since = _.since
        periods = _.periods
        longestPeriod = periods[-1]
        securities = _.securities
        target = securities[0]
        dfs = {}
        
        if (_.MODE == 'TRAIN_OFFLINE'):

            dfs = _.getData(securities, periods, since, None, _.between_time)
            for p in periods:
                _.data[p] =  dfs[p]                

        elif (_.MODE == 'TEST_OFFLINE'):
            
            if( _.data == {} ):                
                dfs = _.getData(securities, periods, since, target, _.between_time )                    
                for p in periods:
                    _.inputDataTest[p] = dfs[p].iloc[ -num :]
                    _.data[p] =  dfs[p].iloc[ : -num ] 
            
            for p in periods:
                row = _.inputDataTest[p].iloc[0]       
                _.inputDataTest[p] = _.inputDataTest[p].iloc[1:]
                _.data[p] = _.data[p].append(row, ignore_index=False)              
            
        elif ( _.MODE == 'TEST_ONLINE' or _.MODE == 'OPERATIONAL'):    
            
            while True:
                logging.info( 'requesting data to the Trading  Platform ...')
                _.tp.HistoryCandleReq(securities, longestPeriod)                
                since = datetime.date.today()
                dfs = _.getData(securities, periods, since, target, _.between_time )
                if (_.isSufficientData(dfs[longestPeriod]) == True):
                    break
                
            for p in periods:
                _.data[p] =  dfs[p]            
        
        else:
            _.log.error('wrong running mode, check self.MODE')


    def trainModel(self, period, params):
        
        msg = 'training&prediction params: '+ period +' '+ str(params)
        logging.info( msg )
        df = self.data[period]
        self.models[period] = self.getTrainingModel(df, params, period)

    
    def storePrediction(self, prediction, period, params):

        self.predictions[period].append(prediction)
   

    def predict( self ):
        
        params = self.ds.getSecurityAlgParams(self.securities[0] )
        
        for p in self.periods:            
            if p not in self.models:
                self.trainModel(p, params)                
            logging.info( 'calling the model ...') 
            pred = self.models[p].predict( self.data[p] )
            self.storePrediction( pred, p, params)
            
   
    def displayPredictions (self):
        
        logging.info( 'plotting a graph ...') 

        for p in self.periods:
            preds = copy.deepcopy(self.predictions[p])
            self.showPrediction( preds , p)    
      
        
    def positionAssestment (self, candlePredList,lastCandle ):
        
        exitPrice = 0.0
        entryPrice = 0.0
        decision = 'no-go'
        currentOpen = lastCandle['currentOpen']
        currentHigh = lastCandle['currentHigh']
        currentLow = lastCandle['currentLow']
        currentClose = lastCandle['currentClose']
        
        currentAverage= (currentOpen+currentHigh+currentLow+currentClose)/4
        self.printPrices = False
        numOfInputCandles=4

        firstcandle  = candlePredList[0]
        secondcandle = candlePredList[1]
        thirdcandle  = candlePredList[2]
        forthcandle  = candlePredList[3]
        
        movAvOpen=(firstcandle['Open']+secondcandle['Open']+thirdcandle['Open']+forthcandle['Open'])/numOfInputCandles
        movAvMax=(firstcandle['High']+secondcandle['High']+thirdcandle['High']+forthcandle['High'])/numOfInputCandles
        movAvMin=(firstcandle['Low']+secondcandle['Low']+thirdcandle['Low']+forthcandle['Low'])/numOfInputCandles
        movAvClose=(firstcandle['Close']+secondcandle['Close']+thirdcandle['Close']+forthcandle['Close'])/numOfInputCandles
        
        firstcandleAvg=(firstcandle['Open']+firstcandle['High']+firstcandle['Low']+firstcandle['Close'])/numOfInputCandles
        secondcandleAvg=(secondcandle['Open']+secondcandle['High']+secondcandle['Low']+secondcandle['Close'])/numOfInputCandles
        thirdcandleAvg=(thirdcandle['Open']+thirdcandle['High']+thirdcandle['Low']+thirdcandle['Close'])/numOfInputCandles
        forthcandleAvg=(forthcandle['Open']+forthcandle['High']+forthcandle['Low']+forthcandle['Close'])/numOfInputCandles


        totalAvg=(firstcandleAvg+secondcandleAvg+thirdcandleAvg+forthcandleAvg)/4

        minDelta=10
        #check the color of the candle
        if (currentClose>currentOpen): #if this blue?
        
            # first check if next avarage  max price if higher then current, assume rise
            if (movAvMax>currentHigh):
                print('It seems the market will grow:')
                #check id its more than delta, if its make sente to enter in this postion to get some money
                # choose entance price with respect to the average min price
                
                
                #entrance price like the avarage of the previos candle doesn not work!!
               #maybe not everytime, maybe somtemis will work
              # MAYBE TAKE CLOSE PRICE OF PREVIOS CANDLE
                entryPrice = currentClose
                deltaForExit=15.0
                #TODO THINK ABOUT OUT PRICE
                exitPrice = entryPrice+deltaForExit
                print('We choose entrance price:' + str(entryPrice))
                print('We set the out price:' + str( exitPrice ))
                decision='long'
        else:
            print('It seems the market will go down..')   
            #the candle is black
            if (movAvClose>currentLow):
                print('It seems the market will go down..')  
                entryPrice=currentClose
                deltaForExit=15.0
                exitPrice = entryPrice - deltaForExit
                decision='short'
  
        return entryPrice, exitPrice, decision
    
    def getPositionAssessmentParams(self,predictions):
        
        myCols = ['predictions','time','Mnemonic','EndPrice' ]
        df_in = pd.DataFrame(columns=myCols)
        
        for p in predictions:
            row = pd.DataFrame({
                'time':                 p.training_set.original_df['CalcDateTime'],
                'Mnemonic':             p.training_set.original_df['Mnemonic'],
                'StartPrice':           p.training_set.original_df['StartPrice'],
                'EndPrice':             p.training_set.original_df['EndPrice'],
                'MinPrice':             p.training_set.original_df['MinPrice'],
                'MaxPrice':             p.training_set.original_df['MaxPrice'] ,
                'open_t+1':             p.predictions[0][0],
                'high_t+1':             p.predictions[0][1],
                'low_t+1':              p.predictions[0][2],
                'close_t+1':            p.predictions[0][3],
                'open_t+2':             p.predictions[0][4],
                'high_t+2':             p.predictions[0][5],
                'low_t+2':              p.predictions[0][6],
                'close_t+2':            p.predictions[0][7],
                'open_t+3':             p.predictions[0][8],
                'high_t+3':             p.predictions[0][9],
                'low_t+3':              p.predictions[0][10],
                'close_t+3':            p.predictions[0][11],
                'open_t+4':             p.predictions[0][12],
                'high_t+4':             p.predictions[0][13],
                'low_t+4':              p.predictions[0][14],
                'close_t+4':            p.predictions[0][15]
                }
            )
            df_in = df_in.append(row)
        df_in['timeDate'] = df_in['time']  
        df_in = df_in.set_index('time')
        df = df_in        
        times = sorted(list(df.index.unique()))
        lastTime = times[-1]    
        
        t1 = lastTime + datetime.timedelta(minutes = 1)
        t2 = lastTime + datetime.timedelta(minutes = 2)
        t3 = lastTime + datetime.timedelta(minutes = 3)
        t4 = lastTime + datetime.timedelta(minutes = 4)

        currentOpen = df.iloc[-1].StartPrice
        currentHigh = df.iloc[-1].MaxPrice
        currentLow = df.iloc[-1].MinPrice
        currentClose = df.iloc[-1].EndPrice

        p = df.loc[lastTime]
        candlePredList = [
            {'Date':t1,'Open':currentOpen+p['open_t+1'],'High':currentHigh + p['high_t+1'],'Low': currentLow+p['low_t+1'],'Close': currentClose+ p['close_t+1'] },
            {'Date':t2,'Open':currentOpen+p['open_t+2'],'High':currentHigh + p['high_t+2'],'Low': currentLow+p['low_t+2'], 'Close': currentClose + p['close_t+2']},
            {'Date':t3,'Open':currentOpen+p['open_t+3'],'High':currentHigh + p['high_t+3'],'Low': currentLow+p['low_t+3'], 'Close': currentClose + p['close_t+3']},
            {'Date':t4,'Open':currentOpen+p['open_t+4'],'High':currentHigh + p['high_t+4'],'Low': currentLow+p['low_t+4'], 'Close':currentClose +p['close_t+4']}
        ]
        
        lastCandle = { 
            'currentOpen' : currentOpen,
            'currentHigh' : currentHigh,
            'currentLow' : currentLow,
            'currentClose' : currentClose                      
        }
        
        return candlePredList,lastCandle

    def evaluatePosition (self):        
                
        longestPeriod = self.periods[-1]
        security = self.target
        board = security['board']
        seccode = security['seccode']
        decimals, marketId =    self.ds.getSecurityInfo (security)
        decimals = int(decimals)
        
        byMarket =              self.params['entryByMarket']
        longPositionMargin =    self.params['longPositionMargin']
        shortPositionMargin =   self.params['shortPositionMargin']
        entryTimeSeconds =      self.params['entryTimeSeconds']        
        quantity =              self.params['positionQuantity']
        k =                     self.params['stopLossCoefficient']
        
        
        stoploss = 0.0
        exitPrice =  0.0
        entryPrice = lastCandle[currentClose]
        
        predictions = copy.deepcopy(self.predictions[longestPeriod])
        candlePredList,lastCandle = self.getPositionAssessmentParams(predictions)
  
        entryPrice, exitPrice, takePosition = self.positionAssestment(candlePredList,lastCandle)

        if takePosition == 'long':
            exitPrice = entryPrice  + longPositionMargin
            stoploss = entryPrice  - k * shortPositionMargin
            
        elif takePosition == 'short':
            exitPrice = entryPrice  - shortPositionMargin
            stoploss = entryPrice  + k * longPositionMargin
            
        else:
            exitPrice = entryPrice
            stoploss = entryPrice
              
        position = tp.Position(
            takePosition, board, seccode, marketId, entryTimeSeconds, 
            quantity, entryPrice, exitPrice , stoploss, decimals, byMarket
        )
        logging.info( 'dolph decides: ' + str(position))    
            
        return position

    def takePosition (self, position):

        action = position.takePosition
        if ( action != 'long' and action != 'short' ):
            logging.info( action + ' position, nothing to do')
            return
        
        if self.MODE == 'OPERATIONAL' :
            logging.info('sending a "' + action +'" to Trading platform ...')
            self.tp.processPosition(position)
            
        
           
    
if __name__== "__main__":

    securities = [] 

    #securities.append( {'board':'FUT', 'seccode':'GZZ0'} )


    securities.append( {'board':'FUT', 'seccode':'SRZ0'} )
    # securities.append( {'board':'FUT', 'seccode':'GDZ0'} ) 
    # securities.append( {'board':'FUT', 'seccode':'SiZ0'} )
    #securities.append( {'board':'FUT', 'seccode':'VBZ0'} )

    # securities.append( {'board':'FUT', 'seccode':'RIZ0'} )
    # # securities.append( {'board':'FUT', 'seccode':'EuZ0'} ) 
    # securities.append( {'board':'FUT', 'seccode':'GMZ0'} )
    # # securities.append( {'board':'FUT', 'seccode':'VBZ0'} )

    # securities.append( {'board':'FUT', 'seccode':'EuZ0'} )
    # securities.append( {'board':'FUT', 'seccode':'BRX0'} )

    dolph = Dolph( securities )

    while True:

        dolph.dataAcquisition()
        dolph.predict()
        dolph.displayPredictions()
        dolph.takePosition( dolph.evaluatePosition() )          

        