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
    
        # MODE := 'TRAIN_OFFLINE' | TEST_OFFLINE' | 'TEST_ONLINE' | 'OPERATIONAL'

        self.MODE = 'TEST_OFFLINE' 
        self.tested = False
        self.numTestSample = 600
        self.since = datetime.date(year=2015,month=3,day=6)
        self.between_time = ('09:00', '18:45')


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
        
        preds = self.predictions[longestPeriod]

        lastPrice = preds[-1].training_set.original_df.iloc[-1]['EndPrice']
        lastPrediction = preds[-1].predictions[-1][-1]
                
        #TODO the rules to choose the takePosition must be studied carefuly
        takePosition = 'no-go'
        if lastPrediction > 0.00:
            takePosition = 'long'
        elif  lastPrediction < 0.00:
            takePosition = 'short'                
        # takePosition = 'long'
        #TODO the rules to choose the takePosition must be studied carefuly
        
        # entryPrice = 0.0
        entryPrice = lastPrice
        if not byMarket:
            entryPrice = lastPrice
       
        exitPrice = 0.0 
        stoploss = 0.0
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

<<<<<<< HEAD
    securities.append( {'board':'FUT', 'seccode':'SRZ0'} )
    # securities.append( {'board':'FUT', 'seccode':'GDZ0'} ) 
    # securities.append( {'board':'FUT', 'seccode':'SiZ0'} )
    securities.append( {'board':'FUT', 'seccode':'VBZ0'} )
=======
    # securities.append( {'board':'FUT', 'seccode':'RIZ0'} )
    # # securities.append( {'board':'FUT', 'seccode':'EuZ0'} ) 
    # securities.append( {'board':'FUT', 'seccode':'GMZ0'} )
    # # securities.append( {'board':'FUT', 'seccode':'VBZ0'} )
>>>>>>> 811c04b3e1d7337bfdcf1a49b0f88aa397c05908

    # securities.append( {'board':'FUT', 'seccode':'EuZ0'} )
    # securities.append( {'board':'FUT', 'seccode':'BRX0'} )

    dolph = Dolph( securities )

    while True:

        dolph.dataAcquisition()
        dolph.predict()
        dolph.displayPredictions()
        # dolph.takePosition( dolph.evaluatePosition() )          

        