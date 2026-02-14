# TradingPlatform.py
import json
import asyncio
from abc import ABC, abstractmethod
import logging
import datetime
from datetime import timezone
import time
import pytz
import calendar
import copy
from threading import Thread, Lock
from TradingPlatforms.transaq_connector import structures as ts
from TradingPlatforms.transaq_connector import commands as tc
import alpaca_trade_api as tradeapi
from ib_insync import IB, Trade, Stock, util, LimitOrder, StopOrder, MarketOrder, Order
from DataManagement import DataServer as ds
from Configuration import Conf as cm
from TradingPlatforms.Alpaca.Alpaca_Order import OrderAlpaca
from TradingPlatforms.InteractiveBrokers.OrderIB import OrderIB  # Assuming OrderIB will handle casting
import pandas as pd

log = logging.getLogger("TradingPlatform")

class SimpleTrade:
    def __init__(self, order_id, status):
        self.id = order_id
        self.status = status

    def __repr__(self):
        return f"SimpleTrade(id={self.id}, status='{self.status}')"

class Position:
    
    def __init__(self, takePosition, board, seccode, marketId, 
                 quantity, entryPrice, exitPrice, stoploss, 
                 decimals, client, exitTime=None, correction=None, spread=None, bymarket = False, 
                 entry_id=None, exit_id=None, exit_order_no=None , union = None, 
                 expdate = None, buysell = None , exitOrderRequested = None, exitOrderAlreadyCancelled = None) :
        
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
        # quantity := number of lots
        self.quantity = quantity
        # entryPrice := price you want to buy(sell) if it's a long(short)
        self.entryPrice = entryPrice
        # exitPrice := price you want to re-sell(buy) if it's a long(short)
        self.exitPrice = exitPrice
        self.decimals = decimals
        # stoploss := price you want to exit if your prediction was wrong        
        self.stoploss = stoploss
        self.exitOrderRequested = False if exit_id is None else True
        # to be assigned when position is being proccessed by Tradingplatform
        self.buysell = buysell if buysell else None
        # exitTime := time for a emergency exit, close current position at 
        # this time by market if the planned exit is not executed yet
        self.exitTime = exitTime if exitTime else datetime.datetime.now()   # Default to current time if None
        
        self.correction = correction
        self.spread = spread               
        self.bymarket = bymarket
        
        self.client = client
        self.union = None
        self.expdate = None
        # exit_order_no := it's the number Transaq gives to the order which is 
        # automatically triggered by a tp_executed or sl_executed 
        self.exit_order_no = exit_order_no  # Add exit_order_no field
        self.exitOrderAlreadyCancelled = exitOrderAlreadyCancelled if exitOrderAlreadyCancelled else False

    def __str__(self):
        
        #fmt = "%d.%m.%Y %H:%M:%S"
        msg = ' takePosition='+ self.takePosition 
        msg += ' seccode=' + self.seccode
        msg += ' quantity=' + str(self.quantity)
        msg += ' entryPrice=' + str(self.entryPrice)
        msg += ' exitPrice=' + str(self.exitPrice)
        msg += ' stoploss=' + str(self.stoploss)
        msg += ' decimals=' + str(self.decimals)
        msg += ' entry_id=' + str(self.entry_id)
        msg += ' exit_id=' + str(self.exit_id)
        msg += ' exit_order_no=' + str(self.exit_order_no)
        #msg += ' exitTime=' + str(self.exitTime.strftime(fmt))
        
        return msg


def initTradingPlatform( onCounterPosition ):
           
    #platform  = ds.DataServer().getPlatformDetails(cm.securities)
    platform = cm.platform
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
        self.monitoredExitOrders = []
        self.counterPositions = []
        self.profitBalance = 0
        self.currentTradingHour = 0
        self.candlesUpdateThread = None
        self.candlesUpdateTask = None
        self.fmt = "%d.%m.%Y %H:%M:%S"
        
        self.loadMonitoredPositions() 

    def _init_configuration(self):
        
        self.connected = False
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
        #platform  = self.ds.getPlatformDetails(cm.securities)    
        platform = cm.platform
        self.secrets = platform["secrets"] 
        self.connectOnInit = self.MODE in ['OPERATIONAL', 'TEST_ONLINE']
        self._init_securities()

    def _init_securities(self):
                    
        for sec in self.securities:
            sec['id'] = self.ds.getSecurityIdSQL(sec['board'], sec['seccode'])   

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
        pass

    @abstractmethod
    def cancel_order(self, order_id):
        pass

    @abstractmethod
    def newExitOrder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        pass

    @abstractmethod
    def cancelExitOrder(self, exitOrderId):
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
    def closeExit(self, mp, meo):  
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
    
    @abstractmethod
    def get_net_balance(self):
        pass
    
    @abstractmethod
    def getClientId(self):
        pass
        
    @abstractmethod
    def isMarketOpen(self, seccode):
        pass

    ################ Common methods    #######################################
    
    def get_PositionsByCode (self, seccode) :
        """ common """
        positions = [p for p in self.monitoredPositions if p.seccode == seccode ]

        return positions 

    def storeMonitoredPositions(self):
        """ common 
        This method stores each monitored position in the database as a JSON string.
        """
        def default_serializer(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()  # Convert datetime to ISO format
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            positions_json = [json.dumps(mp.__dict__, default=default_serializer) for mp in self.monitoredPositions]
            clientID = self.getClientId()
            self.ds.store_positions_to_db(positions_json, clientID )   # Save the list of positions as JSON in the database
            
        except Exception as e:
            log.error(f"Failed to store monitored positions: {e}")
        log.info("Monitored positions have been stored successfully.")
    

    def loadMonitoredPositions(self):
        """ common 
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
    
                # Filter positions based on client
                #if 'client' in pos_dict and pos_dict.get('client') == self.secrets['api_key']:
                if 'client' in pos_dict and pos_dict.get('client') == self.getClientId() :
                    # Construct Position object and append to monitoredPositions
                    pos = Position(**pos_dict)
                    self.monitoredPositions.append(pos)
               
        except Exception as e:
            log.error(f"Failed to load monitored positions: {e}")
        
        log.info("Monitored positions have been loaded successfully")
        for mp in self.monitoredPositions:
            log.info(str(mp))
            


    def processPosition(self, position):
        """ common """
        self.reportCurrentOpenPositions()
        log.info(str(position))
        
        if not self.processingCheck(position):
            return       
        
        log.debug("takePosition ...")

        if position.takePosition in ["long", "short"]:   
            log.debug("takePosition ... long oder short ")

            self.openPosition(position)
        elif position.takePosition == "close":
            self.closePosition(position)
        elif position.takePosition == "close-counterPosition":
            self.closePosition(position, withCounterPosition=True)
        else:
            log.error("takePosition must be either long, short or close")
            raise Exception(position.takePosition)
        
        self.reportCurrentOpenPositions()

    
    def cancellAllOrders(self):
        """ common """
        for mo in self.monitoredOrders:
            res = self.cancel_order(mo.id)
            log.debug(repr(res))
        log.debug('finished!')
          
        
    def addClientAccount(self, clientAccount):
        """ common """
        self.clientAccounts.append(clientAccount)
    
    
    def getClientIdByMarket(self, marketId):        
        """ common """
        for c in self.clientAccounts:
            if c.market == marketId:
                return c.id
        raise Exception("market "+str(marketId)+" not found")     


    def getUnionIdByMarket(self, marketId):        
        """ common """
        for c in self.clientAccounts:
            if c.market == marketId:
                return c.union
        raise Exception("market "+marketId+" not found") 

        
    def triggerWhenMatched(self, order):
        """ common """      
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
    
    
    def triggerExitByMarket(self, stopOrder, monitoredPosition):
        """ common """
                
        if monitoredPosition.exit_id != stopOrder.id: 
            return
        
        mp = monitoredPosition
        mp.exitOrderRequested = True
        logging.info(f"trigerring exit by Market {mp.exit_id} due to cancelling {mp}")  

        res = self.new_order(
            mp.board, mp.seccode, mp.client, mp.union, stopOrder.buysell, mp.expdate, 
            mp.quantity, price=mp.entryPrice, bymarket=True, usecredit=False
        )
        if res is None:
            logging.info("Failed to create order by Market for the exit")  
            
        if res.status in cm.statusOrderForwarding or res.status in cm.statusOrderExecuted:
    
            if stopOrder in self.monitoredExitOrders:
                self.monitoredExitOrders.remove(stopOrder)
 
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != stopOrder.id] 

        
    def processEntryOrderStatus(self, order):
        """ common """
        logging.debug(str(order))  
        # clone = {'id': order.id, 'status': order.status} ;  self.triggerWhenMatched(clone) if s in cm.statusOrderExecuted   
        s = order.status
        try:                
            monitoredPosition = self.getPositionByOrder(order)                        

            if s in cm.statusOrderExecuted : 
                
                if monitoredPosition is None:                    
                    if order in self.monitoredOrders:
                        self.monitoredOrders.remove(order)
                        logging.info(f'already processed before, deleting: {repr(order.id)}')
                    
                elif not monitoredPosition.exitOrderRequested:
                    logging.info(f'Order is Filled-Monitored wo exitOrderRequested: {repr(order.id)}')
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
            log.error(f"Failed to processEntryOrderStatus: {e}")
      

    def processExitOrderStatus(self, stopOrder):
        """common"""        
        logging.debug(str(stopOrder))       
        s = stopOrder.status
        m = ''
        try:               
            if s in cm.statusOrderForwarding:
                
                if stopOrder not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(stopOrder)
                    m = f'stopOrder {stopOrder.id} with status: {s} added to monitoredExitOrders'
                    
            elif s in cm.statusExitOrderExecuted :  
                
                self.set_exit_order_no_to_MonitoredPosition(stopOrder)                      
                if stopOrder in self.monitoredExitOrders:
                    self.monitoredExitOrders.remove(stopOrder)
                    m = f'stopOrder: {stopOrder.id} in status: {s} deleted from monitoredExitOrders'
                
            elif s in cm.statusExitOrderFilled :
                
                self.removeMonitoredPositionByExit(stopOrder)
                if stopOrder in self.monitoredExitOrders:
                    self.monitoredExitOrders.remove(stopOrder)
                    m = f'stopOrder: {stopOrder.id} in status: {s} deleted from monitoredExitOrders'                
            
            elif s in cm.statusOrderCanceled:

                monitoredPosition = self.getPositionByOrder(stopOrder)  
                if monitoredPosition is not None:
                    self.triggerExitByMarket(stopOrder, monitoredPosition)                    
                    m = f'Exit Order {monitoredPosition.exit_id} due to cancelling {monitoredPosition}'
               
            else:
                logging.debug(f'status: {s} skipped, belongs to: {cm.statusOrderOthers}')
            
            if m != "":
                logging.info(m)
                    
        except Exception as e:
            log.error(f"Failed to processExitOrderStatus: {e}")
        
        
    def isPositionOpen(self, seccode):
        """common"""       
        inMP = any(mp.seccode == seccode for mp in self.monitoredPositions)
        inMSP = any(msp.seccode == seccode for msp in self.monitoredExitOrders)
        isOrderActive = any(mo.seccode == seccode for mo in self.monitoredOrders)
        
        flag =  inMP or inMSP or isOrderActive         
        return flag


    def reportCurrentOpenPositions(self):
        """common"""        
        numMonPosition = len(self.monitoredPositions)
        numMonOrder = len(self.monitoredOrders)
        numMonExitOrder = len(self.monitoredExitOrders)
        
        msg = "\n"
        msg += f'monitored Positions : {numMonPosition}\n'
        for mp in self.monitoredPositions:
            msg += str(mp) + '\n'
        msg += f'monitored Entry Orders    : {numMonOrder}\n'
        for mo in self.monitoredOrders:
            msg += str(mo) + '\n'
        msg += f'monitored Exit Orders: {numMonExitOrder}\n'
        for meo in self.monitoredExitOrders:
            msg += str(meo) + '\n'
        
        logging.info(msg)
        total = numMonOrder + numMonExitOrder
        return total
    
    
    def getExpDate(self, seccode):
        """common"""
        tradingPlatformTime = self.getTradingPlatformTime()
        plusNsec = datetime.timedelta( seconds=cm.entryTimeSeconds)
        tradingPlatformTime_plusNsec = tradingPlatformTime + plusNsec
        
        return tradingPlatformTime_plusNsec
    
    
    def closePosition(self, position, withCounterPosition=False):
        """common"""        
        code = position.seccode        
        monitoredPosition = self.getMonitoredPositionBySeccode(code)
        monitoredExitOrder = self.getmonitoredExitOrderBySeccode(code)
        
        if monitoredPosition is None or monitoredExitOrder is None:
            logging.error("position Not found, recheck this case")            
        else:
            log.info('close action received, closing position...')
            self.closeExit(monitoredPosition, monitoredExitOrder)

    
    
    def processingCheck(self, position):
        """common""" 
        self.reportCurrentOpenPositions()

        try:
            if self.MODE != 'OPERATIONAL' :
                m = f'not performing {position.takePosition} because of mode {self.MODE}'
                logging.info(m)
                return False
            
            # if not self.isMarketOpen(position.seccode):
            #     m = f'not performing {position.takePosition} cus the Market is closed for {position.seccode}'
            #     logging.info(m)  
            #     return False            
            
            ct = self.getTradingPlatformTime().time()  
            if not (cm.tradingTimes[0] <= ct <= cm.tradingTimes[1]):
                logging.info(f'We are outside trading hours: {ct}...')  
                return False
            
            ct = self.getTradingPlatformTime()
            if ct.weekday() in [calendar.SATURDAY, calendar.SUNDAY]:
                logging.info('we are on Saturday or Sunday ...')  
                return False
    
            # Only check self.tc if it's relevant, e.g., for platforms that use tc
            if hasattr(self, 'tc') and self.tc is not None and self.tc.connected:
                position.client = self.getClientIdByMarket(position.marketId)
                position.union = self.getUnionIdByMarket(position.marketId)
            
            
            if self.isPositionOpen(position.seccode) and position.takePosition not in ['close', 'close-counterPosition']:
                msg = f'there is a position opened for {position.seccode}'            
                logging.warning(msg)
                return False
            
            if not self.connected:
                msg = 'Trading platform not connected yet ...'            
                logging.warning(msg)            
                return False
            
            logging.info('processing "'+ position.takePosition +'" at Trading platform ...')
            return True
        
        except Exception as e:
            log.error(f"Error : {e}")
            return False
        
 
    def cancelTimedoutEntries(self):
        """common"""        
        list2cancel = []
        nSec = datetime.timedelta( seconds=cm.entryTimeSeconds)
        currentTime = datetime.datetime.now(timezone.utc)
        
        for mp in self.monitoredPositions:
            for mo in self.monitoredOrders:
                if mp.entry_id == mo.id:
                    expTime = mo.time + nSec                
                    if currentTime > expTime:
                        self.cancel_order(mo.id)
                        list2cancel.append(mo)
                        msg = f'Cancelling Order expiring at {expTime.isoformat()}, {mo}'
                        log.info(msg)
                    break
                
        for mo in list2cancel:
            if mo in self.monitoredOrders:
                self.monitoredOrders.remove(mo)
                self.monitoredPositions = [p for p in self.monitoredPositions if p.entry_id != mo.id]
                     

    def cancelTimedoutExits(self):
        """common"""        
        tradingPlatformTime = self.getTradingPlatformTime()
        current_time_only = tradingPlatformTime.time()
        
        for mp in self.monitoredPositions:
            for meo in self.monitoredExitOrders:
                if mp.exit_id == meo.id and current_time_only > cm.time2close and mp.exitOrderAlreadyCancelled == False:
                    log.info(f'time-out exit detected, closing exit for {meo}')
                    self.closeExit(mp, meo)
                    break


    def updatePortfolioPerformance(self, status):
        """common"""        
        if status == 'tp_executed':
            self.profitBalance += 1
        elif status == 'sl_executed':
            self.profitBalance -= 1
        else:
            m = f'status: {status} does not update the portfolio performance'
            logging.info(m)
        logging.info(f'portforlio balance: {self.profitBalance}')       


    def updateTradingHour(self):
        """common"""        
        tradingPlatformTime = self.getTradingPlatformTime()
        currentHour = tradingPlatformTime.hour
        if self.currentTradingHour != currentHour:
            self.currentTradingHour = currentHour
            self.profitBalance = 0
            logging.debug('hour changed ... profitBalance has been reset ')


    def getProfitBalance(self):
        """common"""        
        return self.profitBalance
    
    
    def cancelHangingOrders(self):        
        """common"""        
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
        """common"""        
        monitoredPosition = None
        for mp in self.monitoredPositions:
            if mp.seccode == seccode:
                monitoredPosition = mp
                break
        if monitoredPosition is None:
            logging.debug( "monitoredPosition Not found, recheck this case")
            
        return monitoredPosition
    
    
    def getmonitoredExitOrderBySeccode(self, seccode):
        """common"""        
        monitoredExitOrder = None        
        for meo in self.monitoredExitOrders:
            if meo.seccode == seccode :
                monitoredExitOrder = meo
                break
            
        if monitoredExitOrder is None:
            logging.error( "monitoredExitOrder Not found, recheck this case")
        
        return monitoredExitOrder




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

    def newExitOrder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """Transaq"""
        return self.tc.newExitOrder(board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market)

    def cancelExitOrder(self, stop_order_id):
        """Transaq"""
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
        """Transaq"""
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        return datetime.datetime.now(moscowTimeZone)
    
    
    def onClientAccountRes(self, obj):
        logging.info( repr(obj) )            
        self.addClientAccount(obj)
        
        
    def onClientOrderPacketRes(self, obj):
        logging.debug( repr(obj) )            
        for o in obj.items:
            if isinstance(o, ts.Order):
                self.processEntryOrderStatus(o)
            elif isinstance(o, ts.StopOrder):
                self.processExitOrderStatus(o)
                
            
    def cancellAllStopOrders(self):
        
        if self.tc.connected == True:
            for meo in self.monitoredExitOrders:
                res = self.tc.cancel_takeprofit(meo.id)
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
                 
        res = self.newExitOrder(
            order.board, order.seccode, order.client, buysell, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            monitoredPosition.correction, monitoredPosition.spread, 
            monitoredPosition.bymarket, False 
        )
        log.info(repr(res))
        if res.success:
            monitoredPosition.exitOrderRequested = True
            m = f"stopOrder of order {order.id} successfully requested"
            logging.info(m)
        else:
            monitoredPosition.exitOrderRequested = False
            logging.error("takeprofit hasn't been processed by transaq")      
        
    
    def closeExit(self, mp, meo):
        """ Transaq """
        
        tradingPlatformTime = self.getTradingPlatformTime()
        list2cancel = []
        tid = None
        res = self.cancelExitOrder(meo.id)
        log.debug(repr(res))
        if res.success:
            list2cancel.append(meo)
            localTime = tradingPlatformTime.strftime(self.fmt)
            exitTime = mp.exitTime.strftime(self.fmt)
            msg = f'localTime: {localTime} exit timedouts at: {exitTime} {repr(meo)}'
            log.info(msg)
            res = self.new_order(
                mp.board, mp.seccode, mp.client, mp.union, meo.buysell,
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
        
        for meo in list2cancel:
            if meo in self.monitoredExitOrders:
                self.monitoredExitOrders.remove(meo)
        
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != meo.id] 
        
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
                        self.tp.processEntryOrderStatus(order)  
                    # Process stop or stop-limit orders
                    elif order.type in ['stop', 'stop_limit']:
                        self.tp.processExitOrderStatus(order)  
                    else:
                        log.error(f"Unknown Order type : {order}")                        
            
            except Exception as e:
                log.error(f"Failed to poll order updates: {e}")
            
            self.tp.cancelTimedoutEntries()
            #self.tp.cancelTimedoutExits()   
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
            self.connected = True
                    
            # Ensure that the stream is running
            log.info("Starting the Alpaca stream...")
            if self.barsUpdateTask is not None:
                log.info('candlesUpdateTask was running before ...')
                self.barsUpdateTask.terminate()
                log.info('stopping barUpdateTask ...')
                time.sleep(2)   
    
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
                log.error(f"Asset {seccode} is not shortable.")
                return None    
                
        try:
            log.info(f"Placing {buysell} order for {seccode} at {price}...")
            return self.api.submit_order(**params)
           
        except Exception as e:
            logging.error(f"Failed to place order: {e}")
            logging.exception("Failed to place order")
            return None


    def cancel_order(self, order_id):
        return self.api.cancel_order(order_id)


    def newExitOrder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """Alpaca"""
        return self.api.submit_order(
            symbol=seccode, qty=quantity, 
            side=buysell.lower(), 
            type='stop_limit', 
            time_in_force='gtc', 
            stop_price=trigger_price_tp,
            limit_price=trigger_price_tp
        )


    def cancelExitOrder(self, stop_order_id):
        """Alpaca"""
        return self.api.cancel_order(stop_order_id)


    def getTradingPlatformTime(self):
        """Alpaca"""
        timeZone = pytz.timezone('America/New_York')
        return datetime.datetime.now(timeZone)
    
    
    def get_candles(self, security, since, until, period):
        """Alpaca"""

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
        """ Alpaca """
        
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
        try:
            buysell = ""
            if position.takePosition == "long":           
                buysell = "buy"
            elif position.takePosition == "short":
                buysell = "sell"
              
            log.debug("in openPosition before")
    
            position.expdate = self.getExpDate(position.seccode)
            price = round(position.entryPrice, position.decimals)
            price = "{0:0.{prec}f}".format(price, prec=position.decimals)
            
            log.debug("in openPosition after getExpDate ")    
        
            res = self.new_order(
                position.board, position.seccode, position.client, position.union,
                buysell, position.expdate, position.quantity, price, position.bymarket, False
            )
            if res is None: return
            log.debug(repr(res))
        
            if res.status in cm.statusOrderForwarding :
                position.entry_id = res.id  # Capture the order ID from Alpaca
                self.monitoredPositions.append(position)
                logging.info(f"entry Order placed successfully. Order ID: {res.id}")
            else:
                logging.error(f"Order failed or in invalid state: {res.status}")
                
        
        except Exception as e:
            log.error(f"Failed to get cash balance: {e}")
            return 0


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
         
        res = self.newExitOrder(
            None, order.symbol, None, buysell, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            None, None, None, False 
        )
        log.info(repr(res))
           
        # Alpaca doesn't have a 'success' flag, so let's check the status
        if res.status in ['new', 'pending_new']:
            monitoredPosition.exitOrderRequested = True
            monitoredPosition.exit_id = res.id                
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
            
            logging.info(f"stopOrder {order.id} successfully requested in Alpaca")
        else:
            logging.error(f"stopOrder {order.id} failed in status: {res.status}")
       
        
    
    def closeExit(self, mp, meo):
        """ Alpaca """

        try: 
            res = self.cancelExitOrder(meo.id)
            log.debug(repr(res))
        
        except Exception as e:
            logging.error(f"Error placing closeExit {mp} , {meo}: {e}")
            mp.exitOrderAlreadyCancelled = False
        
        mp.exitOrderAlreadyCancelled = True        
        

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
        self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != order.id]    


    def get_cash_balance (self) :      
        """Alpaca"""
        try:
            # Get account information
            account = self.api.get_account()
            
            # Retrieve and print the cash balance
            cash_balance = float(account.cash)
            
            log.debug(f"Cash balance: ${cash_balance}")
        
            return cash_balance
        
        except Exception as e:
            log.error(f"Failed to get cash balance: {e}")
            return 0
        
    def get_net_balance(self):
        """Alpaca"""
        # Retrieve the net balance (portfolio value) from the Alpaca account.
        try:
            # Get account information
            account = self.api.get_account()
            
            # Retrieve and print the portfolio value (net balance)
            net_balance = float(account.portfolio_value)

            
            log.debug(f"Net balance (portfolio value): ${net_balance}")
            
            return net_balance
        
        except Exception as e:
            log.error(f"Error retrieving net balance: {e}")
            return 0

    def getClientId(self):
        """Alpaca"""
        # Retrieve Client from the Alpaca secrets.
        try:
            # Get account information
            api_key = self.secrets['api_key']
            log.debug(f"Client: ${api_key}")
 
            return api_key
        
        except Exception as e:
            log.error(f"Error retrieving the api_key: {e}")
            return 0
     
        
    def isMarketOpen(self, seccode):
        """Alpaca
        Check if the market is open for a specific security code on Alpaca.

        Args:
            seccode (str): The security code (symbol) to check.

        Returns:
            bool: True if the market is open and the security is tradable, False otherwise.
        """
        try:
            # Check general market status
            clock = self.api.get_clock()
            if not clock.is_open:
                return False

            # Check if the asset is tradable
            asset = self.api.get_asset(seccode)
            return asset.tradable
        except Exception as e:
            log.error(f"Error checking market status for Alpaca and security '{seccode}': {e}")
            return False


##############################################################################

# class IB_eventLoopTask:
    
#     def __init__(self, tp):
#         self._running = True
#         self.tp = tp
#         log.debug('IB_eventLoopTask Thread initialized...')


#     def terminate(self):
#         self._running = False
#         log.debug('thread IB_eventLoopTask terminated...')
        
        
#     def run(self, securities):
#         log.debug('Running thread IB_eventLoopTask...')

        
#         self.tp.ib = IB()
#         self.tp.ib.errorEvent += self.tp.on_error
#         #self.tp,ib.orderStatusEvent += self.onOrderStatus
        
#         # Connect to the IB gateway or TWS
#         log.info('connecting to Interactive Brokers...')            
#         self.tp.ib.connect(self.host, self.port, clientId=self.client_id)
#         self.connected = True  
        
#         log.info("subscribing to Market data...")
#         self.tp.subscribe_to_market_data()        
        
#         log.info("Starting the IB loop...")   
#         self.tp.ib.run()
        
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # try:
        #     loop.run_until_complete(self.tp.ib.run())
        # except Exception as e:
        #     log.error(f"Error in IB loop: {e}")
        # finally:
        #     loop.close()



class IB_OrderStatusTask:

    def __init__(self, tp):

        self._running = True
        self.tp = tp
        log.info('OrderStatusTask Thread initialized...')


    def terminate(self):

        self._running = False
        log.info('thread OrderStatusTask terminated...')

        
    def run(self):

        time.sleep(15)
        log.info('starting ordersStatusUpdate Thread ...')                

        while self._running:
            self.tp.reportCurrentOpenPositions()
            try:
                # Fetch all open trades from IB
                trades = self.tp.ib.trades()
                for trade in trades:
                    order = OrderIB(trade)
                    if order.type in ['MKT']:  # IB uses 'LMT' for limit and 'MKT' for market orders
                        self.tp.processEntryOrderStatus(order)
                    elif order.type in ['STP', 'LMT', 'STP LMT']:
                        self.tp.processExitOrderStatus(order)
                    else:
                        log.error(f"Unknown Order type : {order}")
            except Exception as e:
                log.error(f"Failed to poll order updates: {e}")
            
            self.tp.cancelTimedoutEntries()
            #self.tp.cancelTimedoutExits()           
            
            # Sleep for 5 seconds before the next poll
            time.sleep(5)



# Handling Interactive Brokers (IB) Statuses
#
#     Submitted: The order has been sent to the exchange or venue and is waiting for execution.
#     PreSubmitted: The order is being validated and queued for execution.
#     PendingSubmit: The order is being processed for submission but is not yet submitted.
#     PendingCancel: The order is being processed for cancellation.
#     Cancelled: The order has been canceled and will not be executed.
#     Filled: The order has been completely executed.
#     PartiallyFilled: A part of the order has been executed, but not the entire quantity.
#     Inactive: The order is inactive and not available for execution.
#     Rejected: The order was rejected and will not be executed.
#     Stopped: The order is halted and cannot proceed to execution.
#     PendingReplace: The order is being modified with new parameters.


class IBTradingPlatform(TradingPlatform):

    def __init__(self, onCounterPosition):

        super().__init__(onCounterPosition)
        
        # Optional: Disable lower-level logs from ib_insync specifically
        logging.getLogger('ib_insync').setLevel(logging.ERROR)
        
        self.ib = IB()
        self.ib.errorEvent += self.on_error
        self.ib.orderStatusEvent += self.onOrderStatus

        self.ib_loop = None  # Will hold the asyncio event loop for cross-thread IB calls
        self.eventLoopTask = None
        self.ordersStatusUpdateTask = None
        self.account_number = self.secrets.get("account_number")
        self.host = self.secrets.get("host", "127.0.0.1")
        self.port = self.secrets.get("port", 7497)
        self.client_id = self.secrets.get("client_id", 1)
        self.req_id_to_symbol = {}
        self.symbol_to_bars = {}  # A dictionary to store symbol -> Bars mapping
        self.ib_lock = Lock() 
        
        if self.connectOnInit :
            self.connect()


    def connect(self):
     """ Interactive Brokers """
     try:
         # Connect to the IB gateway or TWS
         log.info('connecting to Interactive Brokers...')
         self.ib.connect(self.host, self.port, clientId=self.client_id)
         self.connected = True

         # Subscribe to market data BEFORE starting the event loop thread
         # (reqHistoricalData uses loop.run_until_complete() which works fine here)
         log.info("subscribing to Market data...")
         self.subscribe_to_market_data()

         log.info("Retrieving and storing initial candles...")
         now = self.getTradingPlatformTime()
         months_ago = now - datetime.timedelta(days=30)

         for sec in self.securities:
             candles = self.get_candles(sec, months_ago, now, period='1Min')
             self.ds.store_candles_from_IB(candles, sec)

         # Get the event loop that ib.connect() created/used
         self.ib_loop = asyncio.get_event_loop()

         # Start event loop in a separate thread, passing the SAME loop.
         # In Python 3.11+, daemon threads don't inherit the main thread's loop,
         # so we must explicitly set it before calling ib.run()
         log.info("Starting event loop in a separate thread for IB ...")
         def run_event_loop():
             asyncio.set_event_loop(self.ib_loop)
             self.ib.run()

         thread = Thread(target=run_event_loop, daemon=True, name="event loop for IB")
         thread.start()

         log.info('Event loop started, waiting 5 seconds...')
         time.sleep(5)

         self.ordersStatusUpdateTask = IB_OrderStatusTask(self)
         t2 = Thread(
             target = self.ordersStatusUpdateTask.run,
             args = ( ),
             name = "IB_OrderStatusTask"
         )
         t2.start()

         return True  # Successful connection

     except Exception as e:
         log.error(f"Failed to connect to IB: {e}")
         return False
      

    def _run_ib(self, coro, timeout=60):
        """Run an IB async coroutine from a non-event-loop thread.

        Once ib.run() is running in the daemon thread, we can no longer use
        loop.run_until_complete(). Instead, schedule the coroutine on the
        running loop via run_coroutine_threadsafe().
        """
        # Verificar que el loop esté disponible y corriendo
        if not self.ib_loop or not self.ib_loop.is_running():
            log.warning("IB event loop not running, cannot execute async operation")
            return None
        
        # Usar lock para evitar llamadas concurrentes
        with self.ib_lock:
            try:
                future = asyncio.run_coroutine_threadsafe(coro, self.ib_loop)
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                log.error(f"IB operation timed out after {timeout} seconds")
                return None
            except Exception as e:
                log.error(f"Error executing IB operation: {e}")
                import traceback
                log.error(traceback.format_exc())
                return None


    def subscribe_to_market_data(self):
        """ Interactive Brokers """

        self.symbol_to_bars = {}  # A dictionary to store symbol -> Bars mapping
    
        for index, security in enumerate(self.securities):
            contract = Stock(security['seccode'], 'SMART', 'USD')
    
            # Request 1-minute bars with streaming updates
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='300 S',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                keepUpToDate=True
            )
    
            # Store the bars object for reference
            self.symbol_to_bars[security['seccode']] = bars
            log.info(f"Subscribed to 1-minute bars for {security['seccode']}.")
    
        # Register callback for live bar updates
        self.ib.barUpdateEvent += self.on_bar_update

      
   

    def on_bar_update(self, bars, hasNewBar):
        """ Interactive Brokers """
        #log.debug (f"DEBUG bars all {bars}")
        
        if not hasNewBar:
            return  # Only process when there is a new bar
    
        # Match the `Bars` object to a symbol using the stored mapping
        symbol = None
        for key, stored_bars in self.symbol_to_bars.items():
            if stored_bars is bars:  # Compare objects
                symbol = key
                break
        
        if not symbol:
            log.error("Symbol could not be resolved for the incoming bars.")
            return
        
        # Process the most recent bar
        bar = bars[-1]
        log.info(f"Received live bar for {symbol}: {bar}")
        updated_data = {
            'timestamp': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }


        # --- Do not store the bar if we are currently outside_trading_time & VOLUME = 0 ---
        current_time = self.getTradingPlatformTime().time()
        outside_trading_time = not ( cm.tradingTimes[0] <= current_time <= cm.tradingTimes[1] )

        if outside_trading_time and bar.volume == 0:
            log.debug(f"Skipping bar for {symbol} because outside trading time and volume=0")
            return
        # ---------------------

        self.ds.store_bar(symbol, updated_data)
           

    def on_historical_data(self, reqId, bar):
        """ Interactive Brokers 
            Callback for historical data updates.
        """
        symbol = self.req_id_to_symbol.get(reqId)
        if not symbol:
            log.error(f"Unknown reqId: {reqId}")
            return
        
        log.info(f"Received bar for {symbol}: {bar}")
        updated_data = {
            'timestamp': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        self.ds.store_bar(symbol, updated_data)



    def on_tick(self, tickers):
        """ Interactive Brokers """
       
        # Convert the timestamp from Unix time (nanoseconds) to a timezone-aware datetime
        #timeZone = self.getTradingPlatformTimeZone()
        
        for ticker in tickers:
            security_code = ticker.contract.symbol
            #timestamp_ns = ticker.time  # Unix time in nanoseconds
            #timestamp_dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9, tz=timeZone)
            updated_data = {
                'timestamp': ticker.time,
                'open': ticker.open,
                'high': ticker.high,
                'low': ticker.low,
                'close': ticker.close,
                'volume': ticker.volume
            }
            log.info(f"Received update for MktData {security_code}: {updated_data}")
            self.ds.store_bar(security_code, updated_data) 
            

    def onOrderStatus(self, trade: Trade):
        """ Interactive Brokers """
        
        order = OrderIB(trade)
        # Process regular or stop orders
        if order.type in ['MKT']:
            self.processEntryOrderStatus(order)
        elif order.type in ['STP', 'LMT']:
            self.processExitOrderStatus(order)        
     
            
    def get_candles(self, security, since, until, period):
        """Interactive Brokers - Fetch historical candles with improved error handling."""
        
        seccode = security['seccode']
        
        # Map the period to IB's duration and barSize settings
        timeframe_mapping = {
            '1Min': ('1 W', '1 min'),
            'hour': ('1 W', '1 hour'),
            'day': ('1 W', '1 day')
        }
        duration, barSize = timeframe_mapping.get(period, ('1 W', '1 min'))
        
        # Convert localized times to UTC
        since_utc = since.astimezone(pytz.utc)
        until_utc = until.astimezone(pytz.utc)
        
        # Format times explicitly in UTC as required by IB API
        end_date_time = until_utc.strftime('%Y%m%d-%H:%M:%S')
        
        # Create a contract for the security
        contract = Stock(seccode, 'SMART', 'USD')
        
        try:
            # Log details of the request
            log.info(f"Fetching candles for {seccode} from {since_utc} to {until_utc}")
            
            # Fetch historical data from IB
            if self.ib_loop and self.ib_loop.is_running():
                bars = self._run_ib(self.ib.reqHistoricalDataAsync(
                    contract, endDateTime=end_date_time, durationStr=duration,
                    barSizeSetting=barSize, whatToShow='TRADES', useRTH=False))

                if bars is None:
                    log.warning(f"Async historical data call failed for {seccode}, trying sync")
                    bars = self.ib.reqHistoricalData(
                        contract, 
                        endDateTime=end_date_time, 
                        durationStr=duration,
                        barSizeSetting=barSize, 
                        whatToShow='TRADES', 
                        useRTH=False
                    )
            else:
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime=end_date_time, durationStr=duration,
                    barSizeSetting=barSize, whatToShow='TRADES', useRTH=False)
            
            # Check for no data returned
            if not bars:
                log.warning(f"No data returned for {seccode}")
                return None
           
            # Convert the bars to a DataFrame
            df = util.df(bars)            
            log.info(f"Fetched {len(df)} rows of candles for {seccode}")
            
            return df
        
        except Exception as e:
            log.error(f"Failed to fetch candles for {seccode}: {e}")
            return pd.DataFrame()

        

    def disconnect(self):
        
        log.info('disconnecting from Interactive Brokers...')
        if self.ib.isConnected():
            self.ib.disconnect()
        
        #self.eventLoopTask.terminate()
        self.ordersStatusUpdateTask.terminate()
        self.storeMonitoredPositions()

    def on_error(self, reqId, errorCode, errorString, contract):
        
        log.error(f"Error: {errorCode}, {errorString}")


    def get_history(self, board, seccode, period, count, reset=True):
        
        contract = Stock(seccode, 'SMART', 'USD')
        end_dt = self.getTradingPlatformTime()
        duration = f'{period * count} D'  # assuming each period is one day
        
        if not self.ib_loop or not self.ib_loop.is_running():
            log.warning("IB event loop not running for get_history")
            return []
        
        bars = self._run_ib(
            self.ib.reqHistoricalDataAsync(
                contract, 
                endDateTime=end_dt, 
                durationStr=duration,
                barSizeSetting='1 day', 
                whatToShow='MIDPOINT', 
                useRTH=True
            ),
            timeout=30  # Timeout de 30 segundos
        )
        
        return bars if bars is not None else []


    def new_order(self, board, seccode, client, union, buysell, expdate, quantity, price, bymarket, usecredit):
        """Interactive Brokers"""
    
        try:
            # Create IB contract for the stock
            contract = Stock(seccode, 'SMART', 'USD')
    
            # Fetch market details for the security
            market_details = self.ib.reqMktData(contract, '', False, False)
            #self.ib.sleep(1)  # Allow time for market data to be retrieved
    
            # Check if the asset is shortable if this is a 'sell' order
            if buysell.lower() == 'sell':
                if not market_details.shortableShares:
                    logging.error(f"Asset {seccode} is not shortable.")
                    return None
    
            # Determine order type: market or limit
            if price == 0 or bymarket:
                order = MarketOrder(buysell, quantity)
            else:
                order = LimitOrder(buysell, quantity, price)

            order.tif = 'GTD'  # Good 'Til Cancel --> Good until date
            order.goodTillDate = expdate.strftime('%Y%m%d-%H:%M:%S')            
            order.account = self.account_number # Use UTC format: 'yyyymmdd-hh:mm:ss'
            
            # Place order via IB API
            trade = self.ib.placeOrder(contract, order)

            # Return a simplified object with only orderId and status
            return SimpleTrade(trade.order.orderId, trade.orderStatus.status)
    
        except Exception as e:
            log.error(f"Error placing order for {seccode}: {e}")
            return None



    def cancel_order(self, order_id):
        self.ib.cancelOrder(order_id)


    def newExitOrder(self, board, seccode, client, buysell, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """ Interactive Brokers """
        try:
            
            contract = Stock(seccode, 'SMART', 'USD') # Create the contract for the stock
    
            stop_price = float(trigger_price_tp) # Determine stop price and limit price
            limit_price = float(trigger_price_tp)  # Use the target price as the limit price
                
            order = Order() # Create a Limit Order
            order.action = buysell.capitalize()  # 'BUY' or 'SELL'
            order.orderType = 'LMT'  # Limit Order
            order.totalQuantity = quantity
            order.auxPrice = stop_price  # The stop (trigger) price
            order.lmtPrice = limit_price  # The limit price
            order.tif = 'GTC'  # Good 'Til Cancel
            order.account = self.account_number
            
            trade = self.ib.placeOrder(contract, order)   # Submit the order via IB API

            return trade
    
        except Exception as e:
            logging.error(f"Error placing stop-limit order for {seccode}: {e}")
            return None


    def cancelExitOrder(self, exitOrderId):
        """ Interactive Brokers """
        try:     
            self.ib.cancelOrder(exitOrderId, '')

        except Exception as e:
            logging.error(f"Error placing cancelExitOrder {exitOrderId}: {e}")
            return None


    def getTradingPlatformTime(self):
        """ Interactive Brokers """
        usEasternTimeZone = pytz.timezone('US/Eastern')
        return datetime.datetime.now(usEasternTimeZone)
    
    
    def cancellAllStopOrders(self):
        pass


    def closeExit(self, mp, meo):
        """ Interactive Brokers """
        
        try: 
            self.cancelExitOrder(meo.order)
        
        except Exception as e:
            logging.error(f"Error placing closeExit {mp} , {meo}: {e}")
            mp.exitOrderAlreadyCancelled = False
        
        mp.exitOrderAlreadyCancelled = True


    
    def getClientId(self):
        """ Interactive Brokers """  
        # Retrieve Client from the Alpaca secrets.        
        try:
            # Get account information           
            account_number = self.secrets['account_number']
            log.debug(f"account_number of the client: ${account_number}")
 
            return account_number
        
        except Exception as e:
            log.error(f"Error retrieving the client_id: {e}")
            return 0

        
    def getPositionByOrder(self, order):
        """ Interactive Brokers """
        
        monitoredPosition = None
        for m in self.monitoredPositions:
            if (
                order.id in [ m.entry_id, m.exit_id ] 
                or m.exit_order_no == order.id 
                or ( m.seccode == order.seccode and m.buysell == order.buysell  )
            ):
               monitoredPosition = m
               break
        return monitoredPosition

    
    def get_cash_balance(self):
        """ Interactive Brokers """
        try:
            # Verificar que el loop esté disponible
            if not self.ib_loop or not self.ib_loop.is_running():
                log.warning("IB event loop not running, cannot get cash balance")
                return 0.0
            
            # Retrieve the account summary con timeout corto
            account_summary = self._run_ib(
                self.ib.accountSummaryAsync(account=self.account_number),
                timeout=10  # Timeout de 10 segundos
            )
            
            if account_summary is None:
                log.error("Failed to get account summary, returning 0.0")
                return 0.0

            # Extract the cash balance
            cash_balance = next(
                (float(item.value) for item in account_summary if item.tag == 'TotalCashValue'), 
                0.0
            )
            log.info(f"Cash balance: ${cash_balance}")
            return cash_balance

        except Exception as e:
            log.error(f"Failed to get cash balance: {e}")
            return 0.0


    def get_net_balance(self):
        """ Interactive Brokers """
        try:
            # Verificar que el loop esté disponible
            if not self.ib_loop or not self.ib_loop.is_running():
                log.warning("IB event loop not running, cannot get net balance")
                return 0.0
            
            # Retrieve the account summary
            account_summary = self._run_ib(
                self.ib.accountSummaryAsync(account=self.account_number),
                timeout=10  # Timeout de 10 segundos
            )
            
            if account_summary is None:
                log.error("Failed to get account summary, returning 0.0")
                return 0.0

            # Extract the net liquidation value
            net_balance = next(
                (float(item.value) for item in account_summary if item.tag == 'NetLiquidation'), 
                0.0
            )
            log.info(f"Net balance (NetLiquidation): ${net_balance}")
            return net_balance

        except Exception as e:
            log.error(f"Error retrieving net balance: {e}")
            return 0.0


    def openPosition(self, position):
        """ Interactive Brokers """

        buysell = "BUY" if position.takePosition == "long" else "SELL"
        position.bymarket = True # for liquid stocks, using MKT for entry is reasonable, slippage is usually 0–1 cent.
        position.expdate = self.getExpDate(position.seccode)
        position.expdate = self.convert_to_utc(position.expdate) 
        price = round(position.entryPrice, position.decimals)
        price = "{0:0.{prec}f}".format(price, prec=position.decimals)
   
        res = self.new_order(
            position.board, position.seccode, position.client, position.union,
            buysell, position.expdate, position.quantity, price, position.bymarket, False
        )

        if res is None:
            log.error("Failed to create order: new_order returned None")            
        
        elif res.status in cm.statusOrderForwarding or res.status in cm.statusOrderExecuted:
        
            position.entry_id = res.id  # Capture the IB order ID
            self.monitoredPositions.append(position)
            log.info(f"orderId: {res.id}, status: {res.status}")
        
        else:
            log.error(f"Order failed or in invalid state: {res.status}")
        

    def removeMonitoredPositionByExit(self, order):
        """ Interactive Brokers """
        self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_id != order.id]   

        
    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Interactive Brokers """
        pass

    
    def triggerStopOrder(self, order, monitoredPosition):
        """ Interactive Brokers """    
        
        logging.info('Triggering stop order...')        
        # Determine buy/sell action based on position type
        buysell = "SELL" if monitoredPosition.takePosition == "long" else "BUY"
        # Format trigger prices
        trigger_price_tp = "{0:0.{prec}f}".format(
            round(monitoredPosition.exitPrice, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        trigger_price_sl = "{0:0.{prec}f}".format(
            round(monitoredPosition.stoploss, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        
        res = self.newExitOrder(
            None, monitoredPosition.seccode, None, buysell, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            None, None, None, False 
        )
        
        if res is None:
            log.error("Failed to create Exit order: newExitOrder returned None")
            
        elif res.orderStatus.status in cm.statusOrderForwarding or res.orderStatus.status in cm.statusOrderExecuted:
                        
            monitoredPosition.exitOrderRequested = True
            monitoredPosition.exit_id = res.order.orderId  # Capture IB order ID
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
                
            log.info(f"Exit order {order.id} successfully in IB OrderId: {res.order.orderId}")
            log.info(repr(res))  
            
        else:
            log.error(f"failed to Exit order for {order.id}, status: {res.orderStatus.status}")

                                                                                     
    def convert_to_utc(self, timezone_aware_datetime):                                   
        """ Interactive Brokers                                                                             
        Converts a timezone-aware datetime to UTC.                                       
                                                                                         
        Args:                                                                            
        - timezone_aware_datetime: A datetime object that is timezone-aware (i.e., has a 
                                                                                         
        Returns:                                                                         
        - A timezone-aware datetime object in UTC.                                       
        """                                                                              
        if timezone_aware_datetime.tzinfo is None:                                       
            raise ValueError("Input datetime must be timezone-aware.")                   
                                                                                         
        # Convert the datetime to UTC                                                    
        utc_datetime = timezone_aware_datetime.astimezone(pytz.utc)                      
                                                                                         
        return utc_datetime   
                                                           
    def _check_ib_ready(self):
        """ Interactive Brokers: Verify IB is connected and event loop is running"""
        try:
            if not self.ib.isConnected():
                log.error("IB is not connected")
                return False
            if not self.ib_loop or not self.ib_loop.is_running():
                log.error("IB event loop is not running")
                return False
            return True
        except Exception as e:
            log.error(f"Error checking IB readiness: {e}")
            return False


    def isMarketOpen(self, seccode):
        """ Interactive Brokers
        Check if the market is open for a specific security code on Interactive Brokers.

        Args:
            seccode (str): The security code (symbol) to check.

        Returns:
            bool: True if the market is open for the security, False otherwise.
        """
        try:
            # Define the contract
            contract = Stock(seccode, "SMART", "USD")

            if not self.ib_loop or not self.ib_loop.is_running():
                log.warning(f"IB event loop not running, cannot check market status for {seccode}")
                return False

            # Get contract details with timeout
            details = self._run_ib(
                self.ib.reqContractDetailsAsync(contract),
                timeout=10  # Timeout de 10 segundos
            )
            
            if not details:
                log.error(f"No contract details found for security '{seccode}'.")
                return False


            # Get the trading hours for the contract
            current_time = self.getTradingPlatformTime().astimezone(pytz.utc)  # Convert to UTC
            for detail in details:
                for session in detail.tradingHours.split(';'):
                    if 'CLOSED' in session:
                        continue  # Skip closed sessions

                    start, end = session.split('-')
                    start_time = datetime.datetime.strptime(start, '%Y%m%d:%H%M').replace(tzinfo=pytz.utc)
                    end_time = datetime.datetime.strptime(end, '%Y%m%d:%H%M').replace(tzinfo=pytz.utc)

                    # Check if the current time is within the session's trading hours
                    if start_time <= current_time <= end_time:
                        return True
            
            log.info(f"detail.tradingHours: {detail.tradingHours} , current_time : {current_time}")

            return False  # Not within any trading session
        except Exception as e:
            log.error(f"Error checking market status for IB and security '{seccode}': {e}")
            return False
                                    


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
     
