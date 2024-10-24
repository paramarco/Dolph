# TradingPlatform.py

import json
import asyncio
from abc import ABC, abstractmethod
import logging
import datetime
import time
import pytz
import copy
from threading import Thread
from TradingPlatforms.transaq_connector import structures as ts
from TradingPlatforms.transaq_connector import commands as tc
import alpaca_trade_api as tradeapi
from ib_insync import IB, Stock, LimitOrder, StopOrder
from DataManagement import DataServer as ds
from Configuration import Conf as cm
from TradingPlatforms.Alpaca.Alpaca_Order import OrderAlpaca

log = logging.getLogger("TradingPlatform")

class Position:
    
    def __init__(self, takePosition, board, seccode, marketId, entryTimeSeconds,
                 quantity, entryPrice, exitPrice, stoploss, 
                 decimals, exitTime=None, correction=None, spread=None, bymarket = False, 
                 entry_id=None, exit_id=None, exit_order_no=None , union = None, 
                 expdate = None,  client = None, buysell = None , stopOrderRequested = None) :
        
        # id:= transactionid of the first order, "your entry" of the Position
        # will be assigned once upon successful entry of the Position
        self.entry_id = entry_id  # Add entry_id field
        # will be assigned once upon successful entry of the Position
        self.exit_id = exit_id    # Add exit_id field
        # takePosition:= long | short | no-go
        self.takePosition = takePosition
        self.board = board
        self.seccode = seccode
        self.marketId = marketId
        # entryTimeSeconds := cancel position entry if it isn't executed 
        # yet within this seconds 
        self.entryTimeSeconds = entryTimeSeconds
        # quantity := number of lots
        self.quantity = quantity
        # entryPrice := price you want to buy(sell) if it's a long(short)
        self.entryPrice = entryPrice
        # exitPrice := price you want to re-sell(buy) if it's a long(short)
        self.exitPrice = exitPrice
        self.decimals = decimals
        # stoploss := price you want to exit if your prediction was wrong        
        self.stoploss = stoploss
        self.stopOrderRequested = False if exit_id is None else True
        # to be assigned when position is being proccessed by Tradingplatform
        self.buysell = buysell if buysell else None
        # exitTime := time for a emergency exit, close current position at 
        # this time by market if the planned exit is not executed yet
        self.exitTime = exitTime if exitTime else datetime.datetime.now()   # Default to current time if None
        
        self.correction = correction
        self.spread = spread               
        self.bymarket = bymarket
        
        self.client = None
        self.union = None
        self.expdate = None
        # exit_order_no := it's the number Transaq gives to the order which is 
        # automatically triggered by a tp_executed or sl_executed 
        self.exit_order_no = exit_order_no  # Add exit_order_no field

    def __str__(self):
        
        #fmt = "%d.%m.%Y %H:%M:%S"
        msg = ' takePosition='+ self.takePosition 
        msg += ' board=' + self.board 
        msg += ' seccode=' + self.seccode
        msg += ' marketId=' + str(self.marketId)
        msg += ' entryTimeSeconds=' + str(self.entryTimeSeconds)
        msg += ' quantity=' + str(self.quantity)
        msg += ' entryPrice=' + str(self.entryPrice)
        msg += ' exitPrice=' + str(self.exitPrice)
        msg += ' stoploss=' + str(self.stoploss)
        msg += ' decimals=' + str(self.decimals)
        msg += ' bymarket=' + str(self.bymarket)
        msg += ' entry_id=' + str(self.entry_id)
        msg += ' exit_id=' + str(self.exit_id)
        msg += ' exit_order_no=' + str(self.exit_order_no)
        #msg += ' exitTime=' + str(self.exitTime.strftime(fmt))
        
        return msg


def initTradingPlatform( onCounterPosition ):
           
    platform  = ds.DataServer().getPlatformDetails(cm.securities)
    logging.debug(str(platform))

    if platform is None :
        return None
    elif platform["name"] == 'finam':
        return FinamTradingPlatform( onCounterPosition )
    elif platform["name"] == 'alpaca':
        return AlpacaTradingPlatform( onCounterPosition )
    elif platform["name"] == 'interactive_brokers':
        return IBTradingPlatform( onCounterPosition )
    else:
        raise ValueError("Unsupported trading platform")

class TradingPlatform(ABC):
    
    def __init__(self, onCounterPosition ):
        
        self._init_configuration()
        self.onCounterPosition = onCounterPosition
        self.clientAccounts = []
        self.monitoredPositions = []
        self.monitoredOrders = []
        self.monitoredStopOrders = []
        self.counterPositions = []
        self.profitBalance = 0
        self.currentTradingHour = 0
        self.candlesUpdateThread = None
        self.candlesUpdateTask = None
        self.fmt = "%d.%m.%Y %H:%M:%S"
        
        self.loadMonitoredPositions() 

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
        self.ds = ds.DataServer()
        platform  = self.ds.getPlatformDetails(cm.securities)    
        self.secrets = platform["secrets"] 
        self.connectOnInit = self.MODE in ['OPERATIONAL']


    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_history(self, board, seccode, period, count, reset=True):
        pass

    @abstractmethod
    def new_order(self, board, seccode, client, union, buysell, expdate, quantity, price, bymarket, usecredit):
        """ res.success := True | False 
            res.id = [a transaction id ]"""
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        """ res.success := True | False 
            res.id = [a transaction id ]"""
        pass

    @abstractmethod
    def new_stoporder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """ res.success := True | False 
            res.id = [a transaction id ]"""
        pass

    @abstractmethod
    def cancel_stoploss(self, stop_order_id):
        """ res.success := True | False 
            res.id = [a transaction id ]"""
        pass

    @abstractmethod
    def getTradingPlatformTime(self):
        pass
 
    @abstractmethod
    def cancellAllStopOrders(self):
        pass
        
    
    @abstractmethod   
    def openPosition ( self, position): 
        pass
     
    @abstractmethod
    def triggerStopOrder(self, order, monitoredPosition):
        pass

    @abstractmethod
    def closeExit(self, mp, mso):  
        pass
    
    @abstractmethod
    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        pass
    
    @abstractmethod
    def getPositionByOrder(self, order):
        pass
    
    @abstractmethod
    def removeMonitoredPositionByExit(self, order):
        pass

    @abstractmethod
    def get_cash_balance (self) :
        pass
    

    
    ################ Common methods    #######################################
    
    def get_PositionsByCode (self, seccode) :

        positions = [p for p in self.monitoredPositions if p.seccode == seccode ]

        return positions 

    def storeMonitoredPositions(self):
        """
        This method stores each monitored position in the database as a JSON string.
        """
        def default_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()  # Convert datetime to ISO format
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            positions_json = [json.dumps(mp.__dict__, default=default_serializer) for mp in self.monitoredPositions]
            self.ds.store_positions_to_db(positions_json)  # Save the list of positions as JSON in the database
        except Exception as e:
            log.error(f"Failed to store monitored positions: {e}")
        log.info("Monitored positions have been stored successfully.")
    

    def loadMonitoredPositions(self):
        """
        This method loads previously stored monitored positions from the database and restores them.
        """
        try:
            positions_json = self.ds.load_positions_from_db()  # Load the list of JSON positions from the database
            for pos_data in positions_json:
                # Since pos_data is already a dictionary, we don't need to call json.loads
                if isinstance(pos_data, str):
                    pos_dict = json.loads(pos_data)  # Only necessary if the data is a string
                else:
                    pos_dict = pos_data  # Already a dictionary from the JSONB column
    
                # Parse the 'exitTime' field from string to datetime if it's in string format
                if 'exitTime' in pos_dict and isinstance(pos_dict['exitTime'], str):
                    pos_dict['exitTime'] = datetime.datetime.fromisoformat(pos_dict['exitTime'])
    
                # Construct Position object and append to monitoredPositions
                pos = Position(**pos_dict)
                self.monitoredPositions.append(pos)
        except Exception as e:
            log.error(f"Failed to load monitored positions: {e}")
        
        log.info("Monitored positions have been loaded successfully")
        for mp in self.monitoredPositions:
            log.info(str(mp))
            


    def processPosition(self, position):
     
        self.reportCurrentOpenPositions()
        log.info(str(position))
        
        if not self.processingCheck(position):
            return       
        
        if position.takePosition in ["long", "short"]:        
            self.openPosition(position)
        elif position.takePosition == "close":
            self.closePosition(position)
        elif position.takePosition == "close-counterPosition":
            self.closePosition(position, withCounterPosition=True)
        else:
            logging.error("takePosition must be either long, short or close")
            raise Exception(position.takePosition)
        
        self.reportCurrentOpenPositions()

    
    def cancellAllOrders(self):
        for mo in self.monitoredOrders:
            res = self.cancel_order(mo.id)
            log.debug(repr(res))
        log.debug('finished!')
          
        
    def addClientAccount(self, clientAccount):
        self.clientAccounts.append(clientAccount)
    
    
    def getClientIdByMarket(self, marketId):        
        for c in self.clientAccounts:
            if c.market == marketId:
                return c.id
        raise Exception("market "+str(marketId)+" not found")     


    def getUnionIdByMarket(self, marketId):        
        for c in self.clientAccounts:
            if c.market == marketId:
                return c.union
        raise Exception("market "+marketId+" not found") 

        
    def triggerWhenMatched(self, order):
        
        trigger = False
        transactionId = None
        position = None
        
        logging.debug(repr(order))
        
        for cp in self.counterPositions:
            transactionId, position = cp
            if order.id == transactionId:
                trigger = True
                break

        if trigger:
            m = f"triggering onCounterPosition for {str(position)}"
            logging.info(m)
            position2invert = copy.deepcopy(position)
            self.counterPositions = list(filter(lambda x: x[0] != transactionId, self.counterPositions))            
            self.onCounterPosition(position2invert)
        
        
    def processOrderStatus(self, order):
        """ common """
        #logging.debug(repr(order))  
        # clone = {'id': order.id, 'status': order.status} ;  self.triggerWhenMatched(clone) if s in cm.statusOrderExecuted   
        s = order.status
        try:
            if s in cm.statusOrderExecuted : 
                
                monitoredPosition = self.getPositionByOrder(order)                        
                if monitoredPosition is None:                    
                    if order in self.monitoredOrders:
                        self.monitoredOrders.remove(order)
                        logging.info(f'already processed before, deleting: {repr(order.id)}')
                    
                elif not monitoredPosition.stopOrderRequested:
                    logging.info(f'Order is Filled-Monitored wo stopOrderRequested: {repr(order.id)}')
                    self.triggerStopOrder(order, monitoredPosition)                
                else:
                    self.removeMonitoredPositionByExit(order)
                    if order in self.monitoredOrders:
                        self.monitoredOrders.remove(order)
                        logging.info(f"exit complete: {str(monitoredPosition)}")                                   
                
            elif s in cm.statusOrderForwarding :
                
                if order not in self.monitoredOrders:
                    self.monitoredOrders.append(order)
                    logging.info(f'order {order.id} in status:{s} added to monitoredOrders')   
                    
            elif s in cm.statusOrderCanceled :

                self.monitoredPositions = [p for p in self.monitoredPositions if p.entry_id != order.id]
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order)
                    self.cancel_order(order.id)                
                    logging.info(f'order {order.id} with status: {s} deleted from monitoredOrders')
                
            else:                
                logging.debug(f'order {order.id} in status: {s} ')
           
        except Exception as e:
            log.error(f"Failed to processOrderStatus: {e}")
      

    def processStopOrderStatus(self, stopOrder):
        """common"""        
        #logging.debug(repr(stopOrder))       
        s = stopOrder.status
        m = ''
        
        if s in cm.statusOrderForwarding:
            
            if stopOrder not in self.monitoredStopOrders:
                self.monitoredStopOrders.append(stopOrder)
                m = f'stopOrder {stopOrder.id} with status: {s} added to monitoredStopOrders'
                
        elif s in cm.statusStopOrderExecuted :  
            
            self.set_exit_order_no_to_MonitoredPosition(stopOrder)                      
            if stopOrder in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(stopOrder)
                m = f'stopOrder: {stopOrder.id} in status: {s} deleted from monitoredStopOrders'
            
        elif s in cm.statusStopOrderFilled :
            
            self.removeMonitoredPositionByExit(stopOrder)
            if stopOrder in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(stopOrder)
                m = f'stopOrder: {stopOrder.id} in status: {s} deleted from monitoredStopOrders'
            
        
        elif s in cm.statusOrderCanceled:
            
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != stopOrder.id] 
            if stopOrder in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(stopOrder)
                m = f'id: {stopOrder.id} with status: {s} deleted from monitoredStopOrders'
            
        else:
            logging.debug(f'status: {s} skipped, belongs to: {cm.statusOrderOthers}')
        
        if m != "":
            logging.info(m)

        
        
    def isPositionOpen(self, seccode):
        
        inMP = any(mp.seccode == seccode for mp in self.monitoredPositions)
        inMSP = any(msp.seccode == seccode for msp in self.monitoredStopOrders)
        isOrderActive = any(mo.seccode == seccode for mo in self.monitoredOrders)
        
        flag =  inMP or inMSP or isOrderActive         
        return flag


    def reportCurrentOpenPositions(self):
        
        numMonPosition = len(self.monitoredPositions)
        numMonOrder = len(self.monitoredOrders)
        numMonStopOrder = len(self.monitoredStopOrders)
        
        msg = "\n"
        msg += f'monitored Positions : {numMonPosition}\n'
        for mp in self.monitoredPositions:
            msg += str(mp) + '\n'
        msg += f'monitored Orders    : {numMonOrder}\n'
        for mo in self.monitoredOrders:
            msg += repr(mo) + '\n'
        msg += f'monitored StopOrders: {numMonStopOrder}\n'
        for mso in self.monitoredStopOrders:
            msg += repr(mso) + '\n'
        
        logging.info(msg)
        total = numMonOrder + numMonStopOrder
        return total
    
    
    def getExpDate(self, seccode):
        #nSec = next((sec['params']['ActiveTimeSeconds'] for sec in self.securities if seccode == sec['seccode']), 0)
        #TODO        
        nSec = 1000
        if nSec == 0:
            log.error('this shouldn\'t happen')
            return None
        tradingPlatformTime = self.getTradingPlatformTime()
        tradingPlatformTime_plusNsec = tradingPlatformTime + datetime.timedelta(seconds=nSec)
        return tradingPlatformTime_plusNsec.strftime(self.fmt)
    
    
    def closePosition(self, position, withCounterPosition=False):
        
        code = position.seccode        
        monitoredPosition = self.getMonitoredPositionBySeccode(code)
        monitoredStopOrder = self.getMonitoredStopOrderBySeccode(code)
        
        if monitoredPosition is None or monitoredStopOrder is None:
            logging.error("position Not found, recheck this case")            
        else:
            log.info('close action received, closing position...')
            cloneMP = copy.deepcopy(monitoredPosition)
            tid = self.closeExit(monitoredPosition, monitoredStopOrder)
            if withCounterPosition:
                log.info('adding position to counterPositions ...') 
                cloneMP.stopOrderRequested = False
                cloneMP.entry_id = None
                cloneMP.exit_id = None
                cp = (tid, cloneMP)
                self.counterPositions.append(cp)
    
    
    def processingCheck(self, position):
 
        self.reportCurrentOpenPositions()

        if self.MODE != 'OPERATIONAL' :
            m = 'not performing:"' + position.takePosition +'" because of MODE ...'
            logging.info(m)
            return False
        
        ct = self.getTradingPlatformTime()                            
        if ct.hour in cm.nogoTradingHours:
            logging.info(f'we are in a no-go Trading hour: {ct.hour}...')  
            return False

        # Only check self.tc if it's relevant, e.g., for platforms that use tc
        if hasattr(self, 'tc') and self.tc is not None and self.tc.connected:
            position.client = self.getClientIdByMarket(position.marketId)
            position.union = self.getUnionIdByMarket(position.marketId)
        
        
        if self.isPositionOpen(position.seccode) and position.takePosition not in ['close', 'close-counterPosition']:
            msg = f'there is a position opened for {position.seccode}'            
            logging.warning(msg)
            return False
        
        logging.info('processing "'+ position.takePosition +'" at Trading platform ...')
        return True
    
 
    def cancelTimedoutEntries(self):
        
        tradingPlatformTime = self.getTradingPlatformTime()
        list2cancel = []
        
        for mp in self.monitoredPositions:
            for mo in self.monitoredOrders:
                
                if mp.entry_id == mo.id:
                    nSec = mp.entryTimeSeconds
                    orderTime_plusNsec = mo.time + datetime.timedelta(seconds=nSec)
                
                    if tradingPlatformTime > orderTime_plusNsec:
                        res = self.cancel_order(mo.id)
                        log.debug(repr(res))
                        list2cancel.append(mo)
                        localTime = tradingPlatformTime.strftime(self.fmt)
                        expTime = orderTime_plusNsec.strftime(self.fmt)
                        msg = f'localTime: {localTime} entry timedouts at: {expTime} {repr(mo)}'
                        log.info(msg)
                    break
                
        for mo in list2cancel:
            if mo in self.monitoredOrders:
                self.monitoredOrders.remove(mo)
                self.monitoredPositions = [p for p in self.monitoredPositions if p.entry_id != mo.id]


    def cancelTimedoutExits(self):
        
        tradingPlatformTime = self.getTradingPlatformTime()
        for mp in self.monitoredPositions:
            for mso in self.monitoredStopOrders:
                if mp.exit_id == mso.id and tradingPlatformTime > mp.exitTime:
                    log.info('time-out exit detected, closing exit')
                    self.closeExit(mp, mso)
                    break


    def updatePortfolioPerformance(self, status):
        
        if status == 'tp_executed':
            self.profitBalance += 1
        elif status == 'sl_executed':
            self.profitBalance -= 1
        else:
            m = f'status: {status} does not update the portfolio performance'
            logging.info(m)
        logging.info(f'portforlio balance: {self.profitBalance}')       


    def updateTradingHour(self):
        
        tradingPlatformTime = self.getTradingPlatformTime()
        currentHour = tradingPlatformTime.hour
        if self.currentTradingHour != currentHour:
            self.currentTradingHour = currentHour
            self.profitBalance = 0
            logging.debug('hour changed ... profitBalance has been reset ')


    def getProfitBalance(self):
        return self.profitBalance
    
    
    def cancelHangingOrders(self):        
        
        tradingPlatformTime = self.getTradingPlatformTime()
        list2cancel = []
        for mo in self.monitoredOrders:
            nSec = next((sec['params']['ActiveTimeSeconds'] for sec in self.securities if mo.seccode == sec['seccode']), 0)
            if nSec == 0:
                log.error('this shouldn\'t happen')
                return
            orderTime_plusNsec = mo.time + datetime.timedelta(seconds=nSec)
            if tradingPlatformTime > orderTime_plusNsec:
                list2cancel.append(mo)
                msg = f'Order hanging since: {mo.time.strftime(self.fmt)}'
                log.info(msg)                
        for mo in list2cancel:
            if mo in self.monitoredOrders:
                clone = copy.deepcopy(mo)
                self.monitoredOrders.remove(mo)
                res = self.cancel_order(clone.id)
                if res.success:
                    res = self.new_order(
                        clone.board, clone.seccode, clone.client, clone.union,
                        clone.buysell, self.getExpDate(clone.seccode),
                        clone.quantity, price=0, bymarket=True, usecredit=False
                    )
                    if res.success:
                        log.info('exit was successfully processed' + repr(clone))
                    else:
                        log.error('exit was erroneously processed')
                else:
                    log.error("cancel active-order error by transaq")


    def getMonitoredPositionBySeccode(self, seccode):
        
        monitoredPosition = None
        for mp in self.monitoredPositions:
            if mp.seccode == seccode:
                monitoredPosition = mp
                break
        if monitoredPosition is None:
            logging.debug( "monitoredPosition Not found, recheck this case")
            
        return monitoredPosition
    
    
    def getMonitoredStopOrderBySeccode(self, seccode):
        
        monitoredStopOrder = None        
        for mso in self.monitoredStopOrders:
            if mso.seccode == seccode :
                monitoredStopOrder = mso
                break
            
        if monitoredStopOrder is None:
            logging.error( "monitoredStopOrder Not found, recheck this case")
        
        return monitoredStopOrder




##############################################################################

class candleUpdateTask:
      
    def __init__(self, tc):
        self._running = True
        self.tc = tc
      
    def terminate(self):
        self._running = False
          
    def run(self, securities):
        
        while self._running :            
            for s in securities:                                
                if self.tc.connected == True:
                    res = self.tc.get_history( s['board'], s['seccode'], 1, 2, True)
                    log.debug(repr(res))
                else:
                    log.warning('wait!, not connected to TRANSAQ yet...')            
            time.sleep(1)  


class FinamTradingPlatform(TradingPlatform):
    
    def __init__(self, onCounterPosition):
        super().__init__( onCounterPosition )
        self.tc = tc.TransaqConnector()


    def initialize(self):
        self.tc.initialize("log", 3, self.handle_txml_message)                


    def connect(self):
        
        log.info('connecting to TRANSAQ...')
        user_id = self.secrets.get("user_id")
        password = self.secrets.get("password")
        endpoint = self.secrets.get("endpoint")
        res = self.tc.connect(user_id, password, endpoint)
        log.debug(repr(res))

    def setConnected2Transaq(self):
        
        logging.info('connected to TRANSAQ' )
        
        if self.candlesUpdateTask is not None:
            log.info('candlesUpdateTask was running before ...')
            self.candlesUpdateTask.terminate()
            log.info('stopping candlesUpdateTask ...')
            time.sleep(2)   

        self.tc.connected = True
        log.info('requesting last 300 entries of the securities ...')
        self.HistoryCandleReq( self.securities, 1, 300)                 
        
        log.info('starting candlesUpdateThread ...')
        self.candlesUpdateTask = candleUpdateTask(self.tc)
        t = Thread(
            target = self.candlesUpdateTask.run, 
            args = ( self.securities, )
        )
        t.start()         
        

    def disconnect(self):
        """Transaq"""
        log.info('stopping candlesUpdateTask ...')
        if self.candlesUpdateTask is not None:
            self.candlesUpdateTask.terminate()
            
        log.info('disconnecting from TRANSAQ...')
        if self.tc.connected:
            self.tc.disconnect()
            self.tc.uninitialize()


    def get_history(self, board, seccode, period, count, reset=True):
        
        if self.tc.connected:
            res = self.tc.get_history(board, seccode, period, count, reset)
            log.debug(repr(res))
        else:
            log.warning('wait!, not connected to TRANSAQ yet...')


    def HistoryCandleReq (self, securities, period, count = 2):
        for s in securities:
            self.get_history(s['board'], s['seccode'], 1 , count) 
            
        time.sleep(2) 

    def new_order(self, board, seccode, client, union, buysell, expdate, quantity, price, bymarket, usecredit):
         """Transaq"""
         return self.tc.new_order(board, seccode, client, union, buysell, expdate, union, quantity, price, bymarket, usecredit)

    def cancel_order(self, order_id):
        return self.tc.cancel_order(order_id)

    def new_stoporder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """Transaq"""
        return self.tc.new_stoporder(board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market)

    def cancel_stoploss(self, stop_order_id):
        return self.tc.cancel_stoploss(stop_order_id)

    def handle_txml_message(self, obj):
        if isinstance(obj, ts.CmdResult):
            logging.info(repr(obj))
        elif isinstance(obj, ts.ClientOrderPacket):
            self.onClientOrderPacketRes(obj)
        elif isinstance(obj, ts.ClientAccount):
            self.onClientAccountRes(obj)
        elif isinstance(obj, ts.ServerStatus):
            if obj.connected == 'true':
                self.setConnected()
            else:
                self.disconnect()
                self.tc.connected = False
                while not self.tc.connected:
                    log.info('waiting 60 seconds to establish a connection...')
                    time.sleep(60)
                    self.tc.initialize("log", 3, self.handle_txml_message)
                    self.connect()
        elif isinstance(obj, ts.HistoryCandlePacket):
        
            logging.debug( repr(obj) )            
            self.ds.storeCandles(obj)            
            self.cancelTimedoutEntries()
            self.cancelTimedoutExits()
            self.cancelHangingOrders()
            self.updateTradingHour()
            
        elif isinstance(obj, ts.MarketPacket):
            pass  # logging.info( repr(obj) )

    def setConnected(self):
        log.info('connected to TRANSAQ')
        self.tc.connected = True
        self.HistoryCandleReq(self.securities, 1, 300)
        self.candlesUpdateTask = candleUpdateTask(self.tc)
        t = Thread(target=self.candlesUpdateTask.run, args=(self.securities,))
        t.start()

    def getTradingPlatformTime(self):
       
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        return datetime.datetime.now(moscowTimeZone)
    
    
    def onClientAccountRes(self, obj):
        logging.info( repr(obj) )            
        self.addClientAccount(obj)
        
        
    def onClientOrderPacketRes(self, obj):
        logging.debug( repr(obj) )            
        for o in obj.items:
            if isinstance(o, ts.Order):
                self.processOrderStatus(o)
            elif isinstance(o, ts.StopOrder):
                self.processStopOrderStatus(o)
                
            
    def cancellAllStopOrders(self):
        
        if self.tc.connected == True:
            for mso in self.monitoredStopOrders:
                res = self.tc.cancel_takeprofit(mso.id)
                log.debug(repr(res))
            log.debug('finished!')
        else:
            log.warning('wait!, not connected to TRANSAQ yet...')  
                              
        
    def openPosition ( self, position): 
        """
        TRANSAQ
        """    
        if position.takePosition == "long":           
            position.buysell = "B"
        elif position.takePosition == "short":
            position.buysell = "S"
        
        position.expdate = self.getExpDate(position.seccode)
        price = round(position.entryPrice , position.decimals)
        price = "{0:0.{prec}f}".format(price,prec=position.decimals)        
        
        res = self.new_order(
            position.board,position.seccode,position.client,position.union,
            position.buysell, position.expdate, position.quantity,price, position.bymarket,False
        )
        log.debug(repr(res))
        if res.success == True:
            position.entry_id = res.id
            self.monitoredPositions.append(position)                                
        else:
            logging.error( "position has not been processed by transaq")       
          

    def triggerStopOrder(self, order, monitoredPosition):
        """  TRANSAQ  """
        
        buysell = "S" if monitoredPosition.buysell == "B" else "B"
        
        trigger_price_tp = "{0:0.{prec}f}".format(
            round(monitoredPosition.exitPrice, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        trigger_price_sl = "{0:0.{prec}f}".format(
            round(monitoredPosition.stoploss, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        
        monitoredPosition.quantity = abs(order.quantity - order.balance)
                 
        res = self.new_stoporder(
            order.board, order.seccode, order.client, buysell, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            monitoredPosition.correction, monitoredPosition.spread, 
            monitoredPosition.bymarket, False 
        )
        log.info(repr(res))
        if res.success:
            monitoredPosition.stopOrderRequested = True
            m = f"stopOrder of order {order.id} successfully requested"
            logging.info(m)
        else:
            monitoredPosition.stopOrderRequested = False
            logging.error("takeprofit hasn't been processed by transaq")      
        
    
    def closeExit(self, mp, mso):
        """ Transaq """
        
        tradingPlatformTime = self.getTradingPlatformTime()
        list2cancel = []
        tid = None
        res = self.cancel_stoploss(mso.id)
        log.debug(repr(res))
        if res.success:
            list2cancel.append(mso)
            localTime = tradingPlatformTime.strftime(self.fmt)
            exitTime = mp.exitTime.strftime(self.fmt)
            msg = f'localTime: {localTime} exit timedouts at: {exitTime} {repr(mso)}'
            log.info(msg)
            res = self.new_order(
                mp.board, mp.seccode, mp.client, mp.union, mso.buysell,
                mp.expdate, mp.quantity, price=0, bymarket=True, usecredit=False
               
            )
            log.debug(repr(res))
            if res.success:
                log.info('exit request was successfully processed')
                tid = res.id
            else:
                log.error('exit request was erroneously processed')
        
        else:
            logging.error("cancel stop-order error by transaq")
        
        for mso in list2cancel:
            if mso in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(mso)
        
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != mso.id] 
        
        return tid
        
    
    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Transaq """
        for mp in self.monitoredPositions: #TODO  
            if mp.exit_id == stopOrder.id and stopOrder.order_no is not None: 
                mp.exit_order_no = stopOrder.order_no
                break  
        

    def getPositionByOrder(self, order):
        """Transaq"""
        monitoredPosition = None
        for m in self.monitoredPositions:
            if order.id in [ m.entry_id, m.exit_id ] or  m.exit_order_no == order.order_no:
               monitoredPosition = m
               break
        return monitoredPosition
        
    
    def removeMonitoredPositionByExit(self, order):
        """Transaq"""
        self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_order_no != order.order_no] 
    
    
        
    
    
##############################################################################

class barUpdateTask:
    
    def __init__(self, tp):
        self._running = True
        self.tp = tp
        log.debug('barUpdateTask Thread initialized...')


    def terminate(self):
        self._running = False
        log.debug('thread barUpdateTask terminated...')
        
        
    def run(self, securities):
        log.debug('Running thread barUpdateTask...')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.tp.stream = tradeapi.Stream(self.tp.api_key, self.tp.api_secret, base_url=self.tp.endpoint)

        log.info("Connected to Alpaca's stream")
        
        # Subscribe to 1-minute bars for each security
        for security in securities:
            seccode = security['seccode']
            log.info(f"Subscribing to 1-minute bars for {seccode} ...")
            self.tp.stream.subscribe_bars(self.tp.on_bar, seccode)
        
        log.info("Starting the Alpaca stream loop...")        
        try:
            loop.run_until_complete(self.tp.stream._run_forever())
        except Exception as e:
            log.error(f"Error in Alpaca stream loop: {e}")
        finally:
            loop.close()


class OrderStatusUpdateTask:
    
    def __init__(self, tp):
        self._running = True
        self.tp = tp
        log.debug('OrderStatusUpdateTask Thread initialized...')


    def terminate(self):
        self._running = False
        log.debug('thread OrderStatusUpdateTask terminated...')
        
        
    def run(self):
        """
        Polls order updates every minute. Handles exceptions to prevent
        maximum recursion errors and logs them properly.
        """
        time.sleep(15) 
        while self._running:
            self.tp.reportCurrentOpenPositions()
            try:
                # Fetch open orders or recent orders from Alpaca
                orders = list( self.tp.api.list_orders(status='all') )  # Fetches open orders; you can adjust the status as needed ('open', 'closed', etc.)
             
                for order_data in orders:
                    #order = Order(order_data)
                    order = OrderAlpaca(order_data._raw)
                    # Process regular orders (market/limit)
                    if order.type in ['limit', 'market']:
                        self.tp.processOrderStatus(order)  
                    # Process stop or stop-limit orders
                    elif order.type in ['stop', 'stop_limit']:
                        self.tp.processStopOrderStatus(order)  
                    else:
                        log.error(f"Unknown Order type : {order}")                        
            
            except Exception as e:
                log.error(f"Failed to poll order updates: {e}")
            
            self.tp.reportCurrentOpenPositions()
            # Sleep for 5 seconds before the next poll
            time.sleep(5) 



# Handling Alpaca Statuses the possible statuses for orders are:
#     new: The order is new and waiting for execution.
#     partially_filled: The order has been partially filled.
#     filled: The order has been completely filled.
#     done_for_day: The order will no longer be filled for the day.
#     canceled: The order was canceled before it was filled.
#     expired: The order expired before it could be filled.
#     rejected: The order was rejected by the trading venue.
#     pending_cancel: The order is being canceled.
#     pending_replace: The order is being replaced with a new order.
#     stopped: The order has been stopped.
#     suspended: The order has been suspended and cannot be filled.
#     calculated: The order is being calculated but is not yet open for execution.
#     accepted: The order has been accepted by the exchange or venue.
#     pending_new: The order has been submitted but not yet acknowledged.

class AlpacaTradingPlatform(TradingPlatform):
    
    def __init__(self, onCounterPosition ):
        log.info("Initializing AlpacaTradingPlatform...")
        super().__init__( onCounterPosition )
        self.api = None
        self.stream = None  # Alpaca WebSocket connection
        self.barsUpdateTask = None
        self.stream = None
        self.initialize()

    def initialize(self):        
        log.info("setting up Alpaca API...")
        self.api_key = self.secrets.get("api_key")
        self.api_secret = self.secrets.get("api_secret")
        self.endpoint = self.secrets.get("endpoint")
        
        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url=self.endpoint)

        if self.connectOnInit :
            self.connect()

    def connect(self):
        try:
            log.info("Retrieving and storing initial candles...")        
            now = self.getTradingPlatformTime()
            months_ago = now - datetime.timedelta(days=30)
        
            for sec in self.securities:
                candles = self.get_candles(sec, months_ago, now, period='1Min')
                self.ds.store_candles(candles, sec)
        
            log.info("Initial candle retrieval and storage complete.")
            # Ensure that the stream is running
            log.info("Starting the Alpaca stream...")
            if self.barsUpdateTask is not None:
                log.info('candlesUpdateTask was running before ...')
                self.barsUpdateTask.terminate()
                log.info('stopping barUpdateTask ...')
                time.sleep(2)   
    
            self.connected = True
            log.info('starting barUpdateThread ...')
            self.barsUpdateTask = barUpdateTask(self)
            t = Thread(
                target = self.barsUpdateTask.run, 
                args = ( self.securities, )
            )
            t.start()  
            
            self.ordersStatusUpdateTask = OrderStatusUpdateTask(self)
            t2 = Thread(
                target = self.ordersStatusUpdateTask.run, 
                args = ( )
            )
            t2.start()  
            
            
        except Exception as e:
            log.error(f"Failed to connect to Alpaca: {e}")              
        

    def disconnect(self):
        """Alpaca"""
        logging.info('Disconnecting from Alpaca...')
        if self.stream:
            self.stream.stop()
        self.barsUpdateTask.terminate()
        self.ordersStatusUpdateTask.terminate()
        self.storeMonitoredPositions()
  

    def getTradingPlatformTimeZone(self):
        return pytz.timezone('America/New_York')  # Example: New York timezone

    def get_history(self, board, seccode, period, count, reset=True):
        
        end_dt = self.getTradingPlatformTime()
        start_dt = end_dt - datetime.timedelta(minutes=period * count)
        barset = self.api.get_barset(seccode, 'minute', start=start_dt.isoformat(), end=end_dt.isoformat(), limit=count)
        return barset


    def new_order(self, board, seccode, client, union, buysell, expdate, quantity, price, bymarket, usecredit):
        """Alpaca"""
        
        params = {
            'symbol': seccode,
            'qty': quantity,
            'side': buysell.lower(),  # Use the mapped side
            'type': 'limit',
            'time_in_force': 'gtc'
        }
        
        if price == 0 or bymarket:            
            params['type'] = 'market'  # Set type to market
        else:
            params['limit_price'] = price  # Only set limit_price for limit orders
            
        # Check if the order is a sell or short-sell, and handle short-selling
        if buysell.lower() == 'sell':
            asset = self.api.get_asset(seccode)
            if not asset.shortable:
                logging.error(f"Asset {seccode} is not shortable.")
                return None    
                
        try:
            logging.info(f"Placing {buysell} order for {seccode} at {price}...")
            return self.api.submit_order(**params)
           
        except Exception as e:
            logging.error(f"Failed to place order: {e}")
            logging.exception("Failed to place order")
            return None


    def cancel_order(self, order_id):
        return self.api.cancel_order(order_id)


    def new_stoporder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """Alpaca"""
        return self.api.submit_order(
            symbol=seccode, qty=quantity, 
            side=buysell.lower(), 
            type='stop_limit', 
            time_in_force='gtc', 
            stop_price=trigger_price_tp,
            limit_price=trigger_price_tp
        )


    def cancel_stoploss(self, stop_order_id):
        return self.api.cancel_order(stop_order_id)


    def getTradingPlatformTime(self):
        timeZone = pytz.timezone('America/New_York')
        return datetime.datetime.now(timeZone)
    
    
    def get_candles(self, security, since, until, period):
        
        seccode = security['seccode']
        # Calculate the granularity
        timeframe_mapping = {
            '1Min': tradeapi.TimeFrame.Minute,
            'hour': tradeapi.TimeFrame.Hour,
            'day': tradeapi.TimeFrame.Day
        }
        
        # Get the appropriate timeframe, defaulting to Minute if the period is not recognized
        timeframe = timeframe_mapping.get(period, tradeapi.TimeFrame.Minute)
    
        # Format the datetime objects as required by the API
        since_dt = since.strftime('%Y-%m-%dT%H:%M:%SZ')
        until_dt = until.strftime('%Y-%m-%dT%H:%M:%SZ')
    
        # Retrieve the bars from the API
        barset = self.api.get_bars(
            seccode,
            timeframe,
            start=since_dt,
            end=until_dt,
            adjustment='raw'
        )
    
        # Return the data as a DataFrame or process it further
        return barset.df


    def cancellAllStopOrders(self):
        pass
        
   
    async def on_bar(self, bar):
        """  Asynchronous callback function to handle bar updates. """
        log.info(f"Received a bar update for {bar.symbol}")    
        seccode = bar.symbol
    
        # Convert the timestamp from Unix time (nanoseconds) to a timezone-aware datetime
        timestamp_ns = bar.timestamp  # Unix time in nanoseconds
        timeZone = self.getTradingPlatformTimeZone()
        timestamp_dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9, tz=timeZone)    
        bar_data = {
            'timestamp': timestamp_dt,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
    
        # Store the bar data using the existing store_bar method
        self.ds.store_bar(seccode, bar_data)
        log.info("Bar stored successfully.")


    def openPosition(self, position):
        """ Alpaca """

        buysell = ""
        if position.takePosition == "long":           
            buysell = "buy"
        elif position.takePosition == "short":
            buysell = "sell"
            
        position.expdate = self.getExpDate(position.seccode)
        price = round(position.entryPrice, position.decimals)
        price = "{0:0.{prec}f}".format(price, prec=position.decimals)
    
        res = self.new_order(
            position.board, position.seccode, position.client, position.union,
            buysell, position.expdate, position.quantity, price, position.bymarket, False
        )    
        log.debug(repr(res))
    
        if res.status in cm.statusOrderForwarding :
            position.entry_id = res.id  # Capture the order ID from Alpaca
            self.monitoredPositions.append(position)
            logging.info(f"entry Order placed successfully. Order ID: {res.id}")
        else:
            logging.error(f"Order failed or in invalid state: {res.status}")


    def triggerStopOrder(self, order, monitoredPosition):
        """ Alpaca """
        logging.info('triggering stopOrder...')
        buysell = "sell" if monitoredPosition.takePosition == "long" else "buy"
        
        trigger_price_tp = "{0:0.{prec}f}".format(
            round(monitoredPosition.exitPrice, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        trigger_price_sl = "{0:0.{prec}f}".format(
            round(monitoredPosition.stoploss, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )  
        #TODO order. quatity? balance?
        #monitoredPosition.quantity = abs(order.quantity - order.balance)
        #monitoredPosition.quantity = order.qty

       # def new_stoporder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
       #      """Alpaca"""
       #      return self.api.submit_order(symbol=seccode, qty=quantity, side=buysell.lower(), type='stop', time_in_force='gtc', stop_price=trigger_price_tp)
        
         
        res = self.new_stoporder(
            None, order.symbol, None, buysell, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            None, None, None, False 
        )
        log.info(repr(res))
           
        # Alpaca doesn't have a 'success' flag, so let's check the status
        if res.status in ['new', 'pending_new']:
            monitoredPosition.stopOrderRequested = True
            monitoredPosition.exit_id = res.id                
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
            
            logging.info(f"stopOrder {order.id} successfully requested in Alpaca")
        else:
            logging.error(f"stopOrder {order.id} failed in status: {res.status}")
       
        
    
    def closeExit(self, mp, mso):
        """ Alpaca """
        
        tradingPlatformTime = self.getTradingPlatformTime()
        list2cancel = []
        tid = None
        res = self.cancel_stoploss(mso.id)
        log.debug(repr(res))
        list2cancel.append(mso)
        localTime = tradingPlatformTime.strftime(self.fmt)
        exitTime = mp.exitTime.strftime(self.fmt)
        msg = f'localTime: {localTime} exit timedouts at: {exitTime} {repr(mso)}'
        log.info(msg)
        res = self.new_order(
            mp.board, mp.seccode, mp.client, mp.union, mso.buysell,
            mp.expdate, mp.quantity, price=mp.entryPrice, bymarket=True, usecredit=False
        )            
        log.debug(repr(res))
        tid = res.id        
        for mso in list2cancel:
            if mso in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(mso)
        
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != mso.id] 
        
        return tid
        

    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Alpaca """
        pass
            
    def getPositionByOrder(self, order):
        """ Alpaca """
        monitoredPosition = None
        for m in self.monitoredPositions:
            if order.id in [ m.entry_id, m.exit_id ] :
               monitoredPosition = m
               break
        return monitoredPosition

    def removeMonitoredPositionByExit(self, order):
        """Alpaca"""
        # log.info(f"my order id : {order.id}")
        # log.info(f"my monitoredPositions before: ")
        # for mp in self.monitoredPositions:
        #     log.info(str(mp))
        self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != order.id]                

        # log.info(f"my monitoredPositions after: ")
        # for mp in self.monitoredPositions:
        #     log.info(str(mp))

    def get_cash_balance (self) :
        """Alpaca"""
        # Get account information
        account = self.api.get_account()
        
        # Retrieve and print the cash balance
        cash_balance = float(account.cash)
        log.debug(f"Cash balance: ${cash_balance}")
        return cash_balance

        
##############################################################################

class IBTradingPlatform(TradingPlatform):

    def __init__(self, onCounterPosition):

        super().__init__(onCounterPosition)
        self.ib = IB()
        self.ib.errorEvent += self.on_error

    def initialize(self):
        # IB specific initialization if any
        pass


    def connect(self):

        log.info('connecting to Interactive Brokers...')
        host = self.secrets.get("host", "127.0.0.1")
        port = self.secrets.get("port", 7497)
        client_id = self.secrets.get("client_id", 1)
        
        try:
            # Connect to the IB gateway or TWS
            self.ib.connect(host, port, clientId=client_id)
            log.info("Connected to IB")
            
            # Subscribe to market data for each security
            self.subscribe_to_market_data()
                    
            return True  # Return True to indicate a successful connection

        except Exception as e:
            log.error(f"Failed to connect to IB: {e}")
        

    def subscribe_to_market_data(self):
        
        for security in self.securities:
            contract = Stock(security['seccode'], 'SMART', 'USD')
            self.ib.reqMktData(contract, '', False, False)
            self.ib.pendingTickersEvent += self.on_tick  # Callback for handling market data updates
        

    def on_tick(self, tickers):
        for ticker in tickers:
            security_code = ticker.contract.symbol
            updated_data = {
                'open': ticker.open,
                'high': ticker.high,
                'low': ticker.low,
                'close': ticker.close,
                'volume': ticker.volume,
                'time': ticker.time
            }
            print(f"Received update for {security_code}: {updated_data}")
            self.ds.store_bar(security_code, updated_data)  # Assuming this method exists in DataServer


    def disconnect(self):
        
        log.info('disconnecting from Interactive Brokers...')
        if self.ib.isConnected():
            self.ib.disconnect()


    def on_error(self, reqId, errorCode, errorString, contract):
        
        log.error(f"Error: {errorCode}, {errorString}")


    def get_history(self, board, seccode, period, count, reset=True):
        contract = Stock(seccode, 'SMART', 'USD')
        end_dt = self.getTradingPlatformTime()
        duration = f'{period * count} D'  # assuming each period is one day
        bars = self.ib.reqHistoricalData(contract, endDateTime=end_dt, durationStr=duration, barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True)
        return bars


    def new_order(self, board, seccode, client, union, buysell, expdate, quantity, price, bymarket, usecredit):
        contract = Stock(seccode, 'SMART', 'USD')
        order = LimitOrder(buysell, quantity, price)
        trade = self.ib.placeOrder(contract, order)
        return trade


    def cancel_order(self, order_id):
        self.ib.cancelOrder(order_id)


    def new_stoporder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        contract = Stock(seccode, 'SMART', 'USD')
        stop_price = float(trigger_price_sl) if buysell.lower() == 'sell' else float(trigger_price_tp)
        order = StopOrder(buysell, quantity, stop_price)
        trade = self.ib.placeOrder(contract, order)
        return trade


    def cancel_stoploss(self, stop_order_id):
        self.ib.cancelOrder(stop_order_id)


    def getTradingPlatformTime(self):
        cetTimeZone = pytz.timezone('CET')
        return datetime.datetime.now(cetTimeZone)
    
    
    def cancellAllStopOrders(self):
        pass


if __name__== "__main__":

     since = cm.since
     until = cm.until
     period = '1Min'
     onCounterPosition = None     
     tp = initTradingPlatform( onCounterPosition )
     
     for sec in cm.securities:
         
         seccode = sec['seccode']
         candles = tp.get_candles( seccode, since, until, period)
         tp.ds.store_candles(candles, seccode)        
     
     