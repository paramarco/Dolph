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
                 entry_id=None, exit_tp_id=None, exit_sl_id=None, exit_order_no=None , union = None,
                 expdate = None, buysell = None , exitOrderRequested = None, exitOrderAlreadyCancelled = None,
                 entry_time=None, entry_limit_prices=None, close_retry_count=0) :
        
        # id:= transactionid of the first order, "your entry" of the Position
        # will be assigned once upon successful entry of the Position
        self.entry_id = entry_id  # Add entry_id field
        # will be assigned once upon successful entry of the Position
        self.exit_tp_id = exit_tp_id    # Add exit_tp_id field
        self.exit_sl_id = exit_sl_id    # Add exit_sl_id field
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
        self.exitOrderRequested = False if exit_tp_id is None else True
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
        # entry_time := UTC datetime when the entry order was filled
        if isinstance(entry_time, str):
            self.entry_time = datetime.datetime.fromisoformat(entry_time)
        else:
            self.entry_time = entry_time
        # entry_limit_prices := list of limit entry attempts with price, time, order_id
        self.entry_limit_prices = entry_limit_prices if entry_limit_prices else []
        # close_retry_count := number of times a market close order was cancelled and restored
        self.close_retry_count = close_retry_count

    def __str__(self):
        
        #fmt = "%d.%m.%Y %H:%M:%S"
        msg = ' position='+ str(self.takePosition)  
        msg += ' seccode=' + self.seccode
        msg += ' quantity=' + str(self.quantity)
        msg += ' entryPrice=' + "{0:0.{prec}f}".format(self.entryPrice, prec=self.decimals)
        msg += ' exitTakeProfit=' + "{0:0.{prec}f}".format(self.exitPrice, prec=self.decimals)
        msg += ' exitStopLoss=' + "{0:0.{prec}f}".format(self.stoploss, prec=self.decimals)
        msg += ' entry_id=' + str(self.entry_id)
        msg += ' exit_tp_id=' + str(self.exit_tp_id)
        msg += ' exit_sl_id=' + str(self.exit_sl_id)
        
        return msg


def initTradingPlatform( onCounterPosition ):
           
    #platform  = ds.DataServer().getPlatformDetails(cm.securities)
    platform = cm.platform
    log.debug(str(platform))

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
        self.position_closed_at = {}  # seccode → UTC datetime of last position close
        self.position_entry_filled_at = {}  # seccode → UTC datetime of entry fill
        self.position_scalp_cooldown = {}  # seccode → extended cooldown seconds (set on quick close)
        self.profitBalance = 0
        self.currentTradingHour = 0
        self.candlesUpdateThread = None
        self.candlesUpdateTask = None
        self.fmt = "%d.%m.%Y %H:%M:%S"
        self._market_close_order_ids = set()
        self._market_close_positions = {}  # order_id -> Position, for restoring on cancel

        if self.MODE != 'TEST_OFFLINE':
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
        self.connectOnInit = self.MODE in ['OPERATIONAL', 'TEST_ONLINE', 'INIT_DB']
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
    def triggerExitOrder(self, order, monitoredPosition):
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
                    # Restore in-memory entry_time lookup from persisted Position
                    if pos.entry_time is not None:
                        self.position_entry_filled_at[pos.seccode] = pos.entry_time
                    elif pos.exitOrderRequested:
                        # Position was filled but entry_time wasn't persisted (legacy).
                        # Try to get fill time from IB trade, fallback to now(UTC).
                        fill_time = None
                        try:
                            for trade in self.ib.trades():
                                if trade.order.orderId == pos.entry_id and trade.fills:
                                    fill_time = trade.fills[-1].time
                                    if fill_time.tzinfo is None:
                                        fill_time = fill_time.replace(tzinfo=datetime.timezone.utc)
                                    break
                        except Exception:
                            pass
                        if fill_time is None:
                            fill_time = datetime.datetime.now(datetime.timezone.utc)
                            log.warning(f"entry_time not found for {pos.seccode} (entry_id={pos.entry_id}), using current UTC time")
                        pos.entry_time = fill_time
                        self.position_entry_filled_at[pos.seccode] = fill_time
               
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
        
        log.debug(repr(order))
        
        for cp in self.counterPositions:
            transactionId, position = cp
            if order.id == transactionId:
                trigger = True
                break

        if trigger:
            m = f"triggering onCounterPosition for {str(position)}"
            log.info(m)
            position2invert = copy.deepcopy(position)
            self.counterPositions = list(filter(lambda x: x[0] != transactionId, self.counterPositions))            
            self.onCounterPosition(position2invert)
    
    
    def triggerExitByMarket(self, stopOrder, monitoredPosition):
        """ common """
                
        if monitoredPosition.exit_tp_id != stopOrder.id: 
            return
        
        mp = monitoredPosition
        mp.exitOrderRequested = True
        log.info(f"trigerring exit by Market {mp.exit_tp_id} due to cancelling {mp}")  

        res = self.new_order(
            mp.board, mp.seccode, mp.client, mp.union, stopOrder.buysell, mp.expdate, 
            mp.quantity, price=mp.entryPrice, bymarket=True, usecredit=False
        )
        if res is None:
            log.info("Failed to create order by Market for the exit")  
            
        if res.status in cm.statusOrderForwarding or res.status in cm.statusOrderExecuted:
    
            if stopOrder in self.monitoredExitOrders:
                self.monitoredExitOrders.remove(stopOrder)
 
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_tp_id != stopOrder.id] 


    def updateExistingExitOrder(self, exitOrder):
        """ common """         
        for i, monitored_order in enumerate(self.monitoredExitOrders):
            if monitored_order.id == exitOrder.id:
                # Update the existing order's attributes
                monitored_order.status = exitOrder.status
                monitored_order.type = exitOrder.type
                monitored_order.side = exitOrder.side
                monitored_order._raw = exitOrder._raw
                monitored_order.order = exitOrder.order
                monitored_order.time = exitOrder.time
                
                log.debug(f"Updated existing exitOrder {exitOrder.id} in monitoredExitOrders: {monitored_order}")
                return True
        
        return False


    def processEntryOrderStatus(self, order):
        """ common """
        log.debug(str(order))  
        # clone = {'id': order.id, 'status': order.status} ;  self.triggerWhenMatched(clone) if s in cm.statusOrderExecuted   
        s = order.status
        try:                
            monitoredPosition = self.getPositionByOrder(order)                        

            if s in cm.statusOrderExecuted : 
                
                if monitoredPosition is None:                    
                    if order in self.monitoredOrders:
                        self.monitoredOrders.remove(order)
                        log.info(f'already processed before, deleting: {repr(order.id)}')
                    
                elif not monitoredPosition.exitOrderRequested:
                    log.info(f'Order is Filled-Monitored wo exitOrderRequested: {repr(order.id)}')
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    monitoredPosition.entry_time = now_utc
                    self.position_entry_filled_at[monitoredPosition.seccode] = now_utc
                    self.triggerExitOrder(order, monitoredPosition)                
                else:
                    self.removeMonitoredPositionByExit(order)
                    if order in self.monitoredOrders:
                        self.monitoredOrders.remove(order)
                        log.info(f"exit complete: {str(monitoredPosition)}")                                   
                
            elif s in cm.statusOrderForwarding :
               
                if order not in self.monitoredOrders:
                    self.monitoredOrders.append(order)
                    log.info(f'order {order.id} in status:{s} added to monitoredOrders')   
                    
            elif s in cm.statusOrderCanceled :

                self.monitoredPositions = [p for p in self.monitoredPositions if p.entry_id != order.id]
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order)
                    self.cancel_order(order.id)                
                    log.info(f'order {order.id} with status: {s} deleted from monitoredOrders')
                
            else:                
                log.debug(f'order {order.id} in status: {s} ')
           
        except Exception as e:
            log.error(f"Failed to processEntryOrderStatus: {e}")
      

    def processExitOrderStatus(self, exitOrder):
        """common"""        
        log.debug(str(exitOrder))  
        self.updateExistingExitOrder(exitOrder)     
        s = exitOrder.status
        m = ''
        try:               
            if s in cm.statusOrderForwarding:
                
                if exitOrder not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(exitOrder)
                    m = f'exitOrder {exitOrder.id} with status: {s} added to monitoredExitOrders'
                            
            elif s in cm.statusExitOrderFilled :
                
                self.removeMonitoredPositionByExit(exitOrder)
                if exitOrder in self.monitoredExitOrders:
                    self.monitoredExitOrders.remove(exitOrder)
                    m = f'exitOrder: {exitOrder.id} in status: {s} deleted from monitoredExitOrders'                
            
            elif ( s in cm.statusOrderCanceled or s in cm.statusExitOrderExecuted) :

                if exitOrder in self.monitoredExitOrders:
                    self.monitoredExitOrders.remove(exitOrder)
                    m = f'exitOrder: {exitOrder.id} in status: {s} deleted from monitoredExitOrders' 
               
            else:
                log.debug(f'status: {s} skipped, belongs to: {cm.statusOrderOthers}')
            
            if m != "":
                log.info(m)
                    
        except Exception as e:
            log.error(f"Failed to processExitOrderStatus: {e}")
        
        
    def isPositionOpen(self, seccode):
        """common"""
        # Block trading for securities with no IB permissions (Error 460)
        if hasattr(self, '_disabled_securities') and seccode in self._disabled_securities:
            return True
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
        msg += f'monitored Positions: {numMonPosition}\n'
        for mp in self.monitoredPositions:
            msg += str(mp) + '\n'
        msg += f'monitored Entry Orders: {numMonOrder}\n'
        for mo in self.monitoredOrders:
            msg += str(mo) + '\n'
        msg += f'monitored Exit Orders: {numMonExitOrder}\n'
        for meo in self.monitoredExitOrders:
            msg += str(meo) + '\n'
        
        log.info(msg)
        total = numMonOrder + numMonExitOrder
        return total
    
    
    def getExpDate(self, seccode):
        """common — uses per-security entryTimeSeconds."""
        tradingPlatformTime = self.getTradingPlatformTime()
        sec = next((s for s in self.securities if s['seccode'] == seccode), {})
        entry_timeout = sec['params']['entryTimeSeconds']
        plusNsec = datetime.timedelta(seconds=entry_timeout)
        tradingPlatformTime_plusNsec = tradingPlatformTime + plusNsec

        return tradingPlatformTime_plusNsec
    
    
    def closePosition(self, position, withCounterPosition=False):
        """common"""        
        code = position.seccode        
        monitoredPosition = self.getMonitoredPositionBySeccode(code)
        monitoredExitOrder = self.getmonitoredExitOrderBySeccode(code)
        
        if monitoredPosition is None or monitoredExitOrder is None:
            log.error("position Not found, recheck this case")            
        else:
            log.info('close action received, closing position...')
            self.closeExit(monitoredPosition, monitoredExitOrder)

    
    
    def processingCheck(self, position):
        """common""" 
        self.reportCurrentOpenPositions()

        try:
            if self.MODE != 'OPERATIONAL' :
                m = f'not performing {position.takePosition} because of mode {self.MODE}'
                log.info(m)
                return False
            
            # if not self.isMarketOpen(position.seccode):
            #     m = f'not performing {position.takePosition} cus the Market is closed for {position.seccode}'
            #     logging.info(m)  
            #     return False            
            
            sec = next((s for s in self.securities if s['seccode'] == position.seccode), {})
            sec_tz = pytz.timezone(sec.get('timezone', 'America/New_York'))
            sec_trading_times = sec.get('tradingTimes', cm.tradingTimes)

            ct_utc = datetime.datetime.now(datetime.timezone.utc)
            ct_in_sec_tz = ct_utc.astimezone(sec_tz).time()

            if not (sec_trading_times[0] <= ct_in_sec_tz <= sec_trading_times[1]):
                log.info(f'Outside trading hours for {position.seccode}: {ct_in_sec_tz}')
                return False

            if ct_utc.astimezone(sec_tz).weekday() in [calendar.SATURDAY, calendar.SUNDAY]:
                log.info(f'Weekend for {position.seccode}')
                return False
    
            # Only check self.tc if it's relevant, e.g., for platforms that use tc
            if hasattr(self, 'tc') and self.tc is not None and self.tc.connected:
                position.client = self.getClientIdByMarket(position.marketId)
                position.union = self.getUnionIdByMarket(position.marketId)
            
            
            if self.isPositionOpen(position.seccode) and position.takePosition not in ['close', 'close-counterPosition']:
                msg = f'there is a position opened for {position.seccode}'            
                log.warning(msg)
                return False
            
            if not self.connected:
                msg = 'Trading platform not connected yet ...'            
                log.warning(msg)            
                return False
            
            log.info('processing "'+ position.takePosition +'" at Trading platform ...')
            return True
        
        except Exception as e:
            log.error(f"Error : {e}")
            return False
        
 
    def cancelTimedoutEntries(self):
        """common"""
        orders_to_cancel = []
        currentTime = datetime.datetime.now(timezone.utc)

        for mp in list(self.monitoredPositions):
            for mo in list(self.monitoredOrders):
                if mp.entry_id == mo.id:
                    # Per-security entry timeout
                    sec = next((s for s in self.securities if s['seccode'] == mp.seccode), {})
                    entry_timeout = sec['params']['entryTimeSeconds']
                    nSec = datetime.timedelta(seconds=entry_timeout)
                    expTime = mo.time + nSec
                    if currentTime > expTime:
                        orders_to_cancel.append(mo)
                        log.info(f'Cancelling Order expiring at {expTime.isoformat()}, {mo}')
                    break

        for mo in orders_to_cancel:
            self.cancel_order(mo.id)
            if mo in self.monitoredOrders:
                self.monitoredOrders.remove(mo)
            self.monitoredPositions = [p for p in self.monitoredPositions if p.entry_id != mo.id]
                     

    def getLastPredictionSignal(self, seccode):
        """common - returns the signal of the last prediction for the given security, or None"""
        sec = next((s for s in self.securities if s['seccode'] == seccode), {})
        longestPeriod = self.periods[-1] if self.periods else None
        predictions = sec.get('predictions', {}).get(longestPeriod, []) if longestPeriod else []
        if not predictions:
            return None
        last_pred = predictions[-1]
        return last_pred.get('signal', 'no-go') if isinstance(last_pred, dict) else last_pred


    def isMarketCloseOrder(self, order):
        """common - returns True if the order is a market-close order sent by closeExit (skip processing)"""
        if order.id not in self._market_close_order_ids:
            return False
        if order.status in cm.statusOrderExecuted:
            self._market_close_order_ids.discard(order.id)
            log.info(f'market close order {order.id} finished with status: {order.status}')
        elif order.status in cm.statusOrderCanceled:
            self._market_close_order_ids.discard(order.id)
            # Restore position that was removed prematurely (e.g. market was closed)
            saved = self._market_close_positions.pop(order.id, None)
            if saved is not None:
                saved.close_retry_count = getattr(saved, 'close_retry_count', 0) + 1
                if saved.close_retry_count >= 3:
                    # Stop retrying — market is likely closed; position stays tracked
                    # but won't trigger further close attempts until next session
                    saved.exitOrderAlreadyCancelled = True
                    self.monitoredPositions.append(saved)
                    log.warning(f'market close order {order.id} CANCELLED for {saved.seccode}, '
                                f'retry limit reached ({saved.close_retry_count}x), '
                                f'position parked in monitoredPositions (no further close attempts)')
                else:
                    saved.exitOrderAlreadyCancelled = False
                    self.monitoredPositions.append(saved)
                    log.warning(f'market close order {order.id} CANCELLED for {saved.seccode}, '
                                f'position restored to monitoredPositions '
                                f'(retry {saved.close_retry_count}/3)')
            else:
                log.warning(f'market close order {order.id} cancelled but no saved position to restore')
        return True


    def cancelTimedoutExits(self):
        """common"""
        positions_to_close = []
        defaut_tz = 'America/New_York'
        default_time2close = datetime.time(16, 30)

        for mp in list(self.monitoredPositions):
            if mp.exitOrderAlreadyCancelled:
                continue

            sec = next((s for s in self.securities if s['seccode'] == mp.seccode), {})
            sec_tz = pytz.timezone(sec.get('timezone', defaut_tz ))
            sec_time2close = sec.get('time2close', getattr(cm, 'time2close', default_time2close ))
            exit_timeout = sec['params']['exitTimeSeconds']
            # Skip positions whose entry hasn't been filled yet (no exit orders)
            if not mp.exitOrderRequested:
                continue
            entry_time = mp.entry_time or self.position_entry_filled_at.get(mp.seccode)
            if entry_time is None:
                log.error(f'skipping timeout check for {mp.seccode}: entry_time is None')
                continue            
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            current_time_in_sec_tz = now_utc.astimezone(sec_tz).time()
            seconds_open = (now_utc - entry_time).total_seconds()
            pred_signal = self.getLastPredictionSignal(mp.seccode)
            sec_trading_times = sec.get('tradingTimes', cm.tradingTimes)
            endOfTradingTimes = sec_trading_times[1]

            should_close = False
            reason = ''

            # Condition 1: current time in security's timezone exceeds time2close
            # This is a hard deadline — close regardless of prediction signal
            if current_time_in_sec_tz > sec_time2close:
                should_close = True
                reason = (f'time2close exceeded ({current_time_in_sec_tz} > {sec_time2close} in {sec.get("timezone", defaut_tz)})')

            # Conditions 2 and 3 require a valid prediction signal
            if not should_close and pred_signal is None:
                continue

            # Condition 2: after endOfTradingTimes but before time2close, close if prediction is opposite
            if not should_close and endOfTradingTimes < current_time_in_sec_tz < sec_time2close and pred_signal != mp.takePosition:
                should_close = True
                reason = (f'after endOfTradingTimes ({current_time_in_sec_tz} > {endOfTradingTimes} '
                    f'before {sec_time2close} in {sec.get("timezone", defaut_tz)}) but positioned opposite to current prediction'
                    f' (prediction={pred_signal} != position={mp.takePosition})')

            # Condition 3: position seconds_open > exitTimeSeconds AND last prediction no longer supports direction
            if not should_close and seconds_open > exit_timeout and pred_signal != mp.takePosition:
                should_close = True
                reason = (f'open {seconds_open/60:.0f}min > {exit_timeout/60:.0f}min, '
                    f'prediction={pred_signal} != position={mp.takePosition}')

            # Condition 4: Signal reversed — close position immediately
            # Only triggers when prediction is the OPPOSITE direction (not no-go)
            if not should_close and pred_signal is not None:
                is_opposite = (
                    (mp.takePosition == 'long' and pred_signal == 'short') or
                    (mp.takePosition == 'short' and pred_signal == 'long')
                )
                if is_opposite:
                    should_close = True
                    reason = (
                        f'SIGNAL REVERSED: position={mp.takePosition}, '
                        f'current prediction={pred_signal} — closing immediately'
                    )

            if should_close:
                meo = next((o for o in self.monitoredExitOrders if o.id == mp.exit_tp_id), None)
                if meo is not None:
                    log.info(f'closing position for {mp.seccode}: {reason}')
                    positions_to_close.append((mp, meo))
                elif mp.exit_tp_id is None:
                    log.info(f'closing position for {mp.seccode} (no exit orders): {reason}')
                    positions_to_close.append((mp, None))

        for mp, meo in positions_to_close:
            self.closeExit(mp, meo)


    def updatePortfolioPerformance(self, status):
        """common"""        
        if status == 'tp_executed':
            self.profitBalance += 1
        elif status == 'sl_executed':
            self.profitBalance -= 1
        else:
            m = f'status: {status} does not update the portfolio performance'
            log.info(m)
        log.info(f'portforlio balance: {self.profitBalance}')       


    def updateTradingHour(self):
        """common"""        
        tradingPlatformTime = self.getTradingPlatformTime()
        currentHour = tradingPlatformTime.hour
        if self.currentTradingHour != currentHour:
            self.currentTradingHour = currentHour
            self.profitBalance = 0
            log.debug('hour changed ... profitBalance has been reset ')


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
            log.debug( "monitoredPosition Not found, recheck this case")
            
        return monitoredPosition
    
    
    def getmonitoredExitOrderBySeccode(self, seccode):
        """common"""        
        monitoredExitOrder = None        
        for meo in self.monitoredExitOrders:
            if meo.seccode == seccode :
                monitoredExitOrder = meo
                break
            
        if monitoredExitOrder is None:
            log.error( "monitoredExitOrder Not found, recheck this case")
        
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
        
        log.info('connected to TRANSAQ' )
        
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
            log.info(repr(obj))
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
        
            log.debug( repr(obj) )            
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
        log.info( repr(obj) )            
        self.addClientAccount(obj)
        
        
    def onClientOrderPacketRes(self, obj):
        log.debug( repr(obj) )            
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
          

    def triggerExitOrder(self, order, monitoredPosition):
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
        
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_tp_id != meo.id] 
        
        return tid
        
    
    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Transaq """
        for mp in self.monitoredPositions: #TODO  
            if mp.exit_tp_id == stopOrder.id and stopOrder.order_no is not None: 
                mp.exit_order_no = stopOrder.order_no
                break  
        

    def getPositionByOrder(self, order):
        """Transaq"""
        monitoredPosition = None
        for m in self.monitoredPositions:
            if order.id in [ m.entry_id, m.exit_tp_id ] or  m.exit_order_no == order.order_no:
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
                    order = OrderAlpaca(order_data._raw)
                    if self.tp.isMarketCloseOrder(order):
                        continue
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
            self.tp.cancelTimedoutExits()
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
        if self.MODE != 'TEST_OFFLINE':
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
                log.info(f"entry Order placed successfully. Order ID: {res.id}")
            else:
                log.error(f"Order failed or in invalid state: {res.status}")
                
        
        except Exception as e:
            log.error(f"Failed to get cash balance: {e}")
            return 0


    def triggerExitOrder(self, order, monitoredPosition):
        """ Alpaca """
        log.info('triggering stopOrder...')
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
            monitoredPosition.exit_tp_id = res.id                
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
            
            log.info(f"stopOrder {order.id} successfully requested in Alpaca")
        else:
            log.error(f"stopOrder {order.id} failed in status: {res.status}")
       
        
    
    def closeExit(self, mp, meo):
        """ Alpaca """

        try:
            # Cancel the pending exit order
            res = self.cancelExitOrder(meo.id)
            log.debug(repr(res))

            # Send market order to close the position
            exit_action = "sell" if mp.takePosition == "long" else "buy"
            close_res = self.api.submit_order(
                symbol=mp.seccode, qty=mp.quantity,
                side=exit_action, type='market', time_in_force='gtc'
            )

            if close_res is None:
                log.error(f"Failed to create market order for closing position {mp.seccode}, keeping position in monitoredPositions")
                return

            self._market_close_order_ids.add(close_res.id)
            log.info(f'exit by market successfully processed for {mp.seccode}, close_order_id={close_res.id}')

            # Only clean up monitored structures AFTER confirming the market close order was accepted
            if meo in self.monitoredExitOrders:
                self.monitoredExitOrders.remove(meo)
            self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_tp_id != meo.id]

            mp.exitOrderAlreadyCancelled = True

        except Exception as e:
            log.error(f"Error placing closeExit {mp} , {meo}: {e}")
            mp.exitOrderAlreadyCancelled = False
        

    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Alpaca """
        pass
            
    def getPositionByOrder(self, order):
        """ Alpaca """
        monitoredPosition = None
        for m in self.monitoredPositions:
            if order.id in [ m.entry_id, m.exit_tp_id ] :
               monitoredPosition = m
               break
        return monitoredPosition

    def removeMonitoredPositionByExit(self, order):
        """Alpaca"""
        self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_tp_id != order.id]    


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


class IB_OrderStatusTask:
    """ Interactive Brokers (IB) handling of events"""

    def __init__(self, tp):

        self._running = True
        self.tp = tp
        self._last_orphan_market_check = 0  # epoch timestamp
        log.info('OrderStatusTask Thread initialized...')


    def terminate(self):

        self._running = False
        log.info('thread OrderStatusTask terminated...')

        
    def run(self):
        """ Interactive Brokers (IB) handling of events"""

        time.sleep(15)
        log.info('starting ordersStatusUpdate Thread ...')

        while self._running:
            self.tp.reportCurrentOpenPositions()
            try:
                # Fetch all open trades from IB
                trades = self.tp.ib.trades()
                for trade in trades:
                    order = OrderIB(trade)
                    if self.tp.isMarketCloseOrder(order):
                        continue

                    # Determine if a LMT order is an entry by checking order ID
                    is_entry_position = any(mp.entry_id == order.id for mp in self.tp.monitoredPositions
                                            if not mp.exitOrderRequested)
                    is_entry_monitored = any(mo.id == order.id for mo in self.tp.monitoredOrders)

                    if order.type in ['MKT'] or (order.type == 'LMT' and (is_entry_position or is_entry_monitored)):
                        self.tp.processEntryOrderStatus(order)
                    elif order.type in ['STP', 'LMT', 'STP LMT']:
                        self.tp.processExitOrderStatus(order)
                    else:
                        log.error(f"Unknown Order type : {order}")
            except Exception as e:
                log.error(f"Failed to poll order updates: {e}")

            self.tp.cancelTimedoutEntries()
            self.tp.cancelTimedoutExits()
            self.tp.reconcileOrphanedPositions()
            self._check_positions_with_closed_markets()

            # Sleep for 5 seconds before the next poll
            time.sleep(5)

    def _check_positions_with_closed_markets(self):
        """Check hourly if any IB position has its market closed. If so, switch to TEST_ONLINE."""
        now = time.time()
        if now - self._last_orphan_market_check < 3600:
            return
        self._last_orphan_market_check = now

        try:
            ib_positions = self.tp.ib.positions()
            if not ib_positions:
                return

            default_tz = 'America/New_York'
            default_trading_times = getattr(cm, 'tradingTimes', (datetime.time(9, 30), datetime.time(16, 0)))

            alerts = []
            for pos in ib_positions:
                if pos.position == 0:
                    continue
                seccode = pos.contract.symbol
                sec = next((s for s in self.tp.securities if s['seccode'] == seccode), None)
                if sec is None:
                    continue

                sec_tz = pytz.timezone(sec.get('timezone', default_tz))
                trading_times = sec.get('tradingTimes', default_trading_times)
                now_in_sec_tz = datetime.datetime.now(datetime.timezone.utc).astimezone(sec_tz)
                current_time = now_in_sec_tz.time()

                is_weekend = now_in_sec_tz.weekday() in (calendar.SATURDAY, calendar.SUNDAY)
                sec_time2close = sec.get('time2close', trading_times[1])
                close_dt = datetime.datetime.combine(now_in_sec_tz.date(), sec_time2close) + datetime.timedelta(minutes=3)
                market_end = close_dt.time()
                is_outside_hours = not (trading_times[0] <= current_time <= market_end)

                if is_weekend or is_outside_hours:
                    direction = "LONG" if pos.position > 0 else "SHORT"
                    alerts.append(
                        f"{seccode}: {direction} {abs(pos.position)} shares, "
                        f"market closed ({current_time} in {sec.get('timezone', default_tz)})"
                    )

            if alerts:
                cm.MODE = 'TEST_ONLINE'
                alert_msg = (
                    "\n" + "=" * 80 + "\n"
                    "CRITICAL: POSITIONS WITH CLOSED MARKETS DETECTED!\n"
                    "MODE changed to TEST_ONLINE — NO NEW TRADES WILL BE SENT\n"
                    "Manual intervention required to normalize.\n"
                    + "=" * 80 + "\n"
                    + "\n".join(alerts) + "\n"
                    + "=" * 80
                )
                log.error(alert_msg)
                self._send_orphan_alert_email(alerts)

        except Exception as e:
            log.error(f"Failed to check positions with closed markets: {e}")

    def _send_orphan_alert_email(self, alerts):
        """Send email alert when positions with closed markets are detected."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            from_email = cm.platform['secrets']['from_email']
            from_password = cm.platform['secrets']['from_password']
            to_emails = cm.platform['secrets']['to_emails']

            subject = "DOLPH ALERT: Positions with closed markets detected"
            body = (
                "CRITICAL: Dolph detected open positions in IB with their markets closed.\n"
                "MODE has been changed to TEST_ONLINE — no new trades will be sent.\n\n"
                "Affected positions:\n" + "\n".join(f"  - {a}" for a in alerts) +
                "\n\nManual intervention required to normalize the situation."
            )

            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('instaltic-com.correoseguro.dinaserver.com', 587)
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
            server.quit()
            log.info("Orphan position alert email sent successfully")
        except Exception as e:
            log.error(f"Failed to send orphan alert email: {e}")

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
         months_ago = now - datetime.timedelta(days=cm.numDaysHistCandles)

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

         self.reconcileBrokerPositions()

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


    def _make_stock_contract(self, seccode):
        """Create an IB Stock contract using per-security exchange/currency config."""
        sec = next((s for s in self.securities if s['seccode'] == seccode), {})
        exchange = sec.get('exchange', 'SMART')
        currency = sec.get('currency', 'USD')
        contract = Stock(seccode, exchange, currency)
        primary_exchange = sec.get('primaryExchange')
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        return contract

    def subscribe_to_market_data(self):
        """ Interactive Brokers """

        self.symbol_to_bars = {}  # A dictionary to store symbol -> Bars mapping

        for index, security in enumerate(self.securities):
            contract = Stock(security['seccode'], security.get('exchange', 'SMART'), security.get('currency', 'USD'))
            if security.get('primaryExchange'):
                contract.primaryExchange = security['primaryExchange']
    
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
        sec = next((s for s in self.securities if s['seccode'] == symbol), {})
        sec_tz = pytz.timezone(sec.get('timezone', 'America/New_York'))
        sec_trading_times = sec.get('tradingTimes', cm.tradingTimes)
        current_time_in_sec_tz = datetime.datetime.now(datetime.timezone.utc).astimezone(sec_tz).time()
        outside_trading_time = not (sec_trading_times[0] <= current_time_in_sec_tz <= sec_trading_times[1])

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

        if self.isMarketCloseOrder(order):
            return

        # Process regular or stop orders
        if order.type in ['MKT']:
            self.processEntryOrderStatus(order)
        elif order.type in ['STP', 'LMT', 'STP LMT']:
            self.processExitOrderStatus(order)
     
            
    def get_candles(self, security, since, until, period):
        """Interactive Brokers - Fetch historical candles with improved error handling."""

        seccode = security['seccode']

        # Map period to IB barSize
        barsize_mapping = {
            '1Min': '1 min',
            'hour': '1 hour',
            'day': '1 day'
        }
        barSize = barsize_mapping.get(period, '1 min')

        # Convert localized times to UTC
        since_utc = since.astimezone(pytz.utc)
        until_utc = until.astimezone(pytz.utc)

        # Calculate actual duration and chunk accordingly
        total_days = (until_utc - since_utc).days + 1
        if total_days <= 5:
            chunk_days = total_days
            duration = f'{total_days} D'
        elif total_days <= 30:
            chunk_days = total_days
            duration = '1 M'
        else:
            chunk_days = 30
            duration = '1 M'

        contract = self._make_stock_contract(seccode)
        all_dfs = []
        chunk_end = until_utc
        chunk_num = 0

        log.info(f"Fetching candles for {seccode}: {total_days} days, duration={duration}, chunks of {chunk_days} days")

        while chunk_end > since_utc:
            chunk_num += 1
            end_date_time = chunk_end.strftime('%Y%m%d-%H:%M:%S')
            log.info(f"  {seccode} chunk {chunk_num}: ending {end_date_time}, duration={duration}")

            try:
                if self.ib_loop and self.ib_loop.is_running():
                    bars = self._run_ib(self.ib.reqHistoricalDataAsync(
                        contract, endDateTime=end_date_time, durationStr=duration,
                        barSizeSetting=barSize, whatToShow='TRADES', useRTH=False))

                    if bars is None:
                        log.warning(f"Async historical data call failed for {seccode}, trying sync")
                        bars = self.ib.reqHistoricalData(
                            contract, endDateTime=end_date_time, durationStr=duration,
                            barSizeSetting=barSize, whatToShow='TRADES', useRTH=False)
                else:
                    bars = self.ib.reqHistoricalData(
                        contract, endDateTime=end_date_time, durationStr=duration,
                        barSizeSetting=barSize, whatToShow='TRADES', useRTH=False)

                if bars:
                    df_chunk = util.df(bars)
                    all_dfs.append(df_chunk)
                    log.info(f"  {seccode} chunk {chunk_num}: {len(df_chunk)} rows")
                else:
                    log.warning(f"  {seccode} chunk {chunk_num}: no data")

            except Exception as e:
                log.error(f"Failed to fetch chunk {chunk_num} for {seccode}: {e}")

            chunk_end -= datetime.timedelta(days=chunk_days)
            time.sleep(1)

        if not all_dfs:
            log.warning(f"No data returned for {seccode}")
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        log.info(f"Fetched total {len(df)} rows of candles for {seccode}")
        return df

        

    def disconnect(self):

        log.info('disconnecting from Interactive Brokers...')
        if self.ib.isConnected():
            self.ib.disconnect()

        #self.eventLoopTask.terminate()
        if self.ordersStatusUpdateTask is not None:
            self.ordersStatusUpdateTask.terminate()
        if self.MODE != 'TEST_OFFLINE':
            self.storeMonitoredPositions()

    def on_error(self, reqId, errorCode, errorString, contract):

        log.error(f"Error: {errorCode}, {errorString}")

        # Error 460: No trading permissions — mark security as disabled
        if errorCode == 460:
            # reqId is the orderId; find the position and remove it
            for mp in list(self.monitoredPositions):
                if mp.entry_id == reqId:
                    seccode = mp.seccode
                    if not hasattr(self, '_disabled_securities'):
                        self._disabled_securities = set()
                    self._disabled_securities.add(seccode)
                    self.monitoredPositions.remove(mp)
                    if any(mo.id == reqId for mo in self.monitoredOrders):
                        self.monitoredOrders = [mo for mo in self.monitoredOrders if mo.id != reqId]
                    log.warning(f"Security {seccode} disabled due to Error 460 (No trading permissions) "
                                f"- removed ghost position entry_id={reqId}")
                    break


    def get_history(self, board, seccode, period, count, reset=True):

        contract = self._make_stock_contract(seccode)
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
            contract = self._make_stock_contract(seccode)
    
            # Fetch market details for the security
            market_details = self.ib.reqMktData(contract, '', False, False)
            #self.ib.sleep(1)  # Allow time for market data to be retrieved
    
            # Check if the asset is shortable if this is a 'sell' order
            if buysell.lower() == 'sell':
                if not market_details.shortableShares:
                    log.error(f"Asset {seccode} is not shortable.")
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



    async def _cancel_order_async(self, order_id):
        """Cancel order on IB event loop thread."""
        # ib.cancelOrder() expects an Order object, not an int
        order = None
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                order = trade.order
                break
        if order is None:
            order = Order(orderId=order_id)
        self.ib.cancelOrder(order)

    def cancel_order(self, order_id):
        self._run_ib(self._cancel_order_async(order_id))


    async def _newExitOrder_async(self, seccode, exit_action, quantity, trigger_price_sl, trigger_price_tp):
        """Place bracket exit orders on IB event loop thread."""
        contract = self._make_stock_contract(seccode)
        utcInteger = int(datetime.datetime.now(timezone.utc).timestamp())
        oca_group = f"OCA_{seccode}_{utcInteger}"

        # --- Take Profit (Limit Order) ---
        tp_order = Order()
        tp_order.action = exit_action
        tp_order.orderType = 'LMT'
        tp_order.totalQuantity = quantity
        tp_order.lmtPrice = float(trigger_price_tp)
        tp_order.tif = 'GTC' # Good 'Til Cancel
        tp_order.account = self.account_number

        # --- Stop Loss (Stop Order) ---
        sl_order = Order()
        sl_order.action = exit_action
        sl_order.orderType = 'STP'
        sl_order.totalQuantity = quantity
        sl_order.auxPrice = float(trigger_price_sl)  # trigger price
        sl_order.tif = 'GTC' # Good 'Til Cancel
        sl_order.account = self.account_number

        # --- Link them via OCA group ---
        tp_order.ocaGroup = oca_group
        tp_order.ocaType = 1  # 1 = Cancel remaining orders in group

        sl_order.ocaGroup = oca_group
        sl_order.ocaType = 1

        # Place both orders
        tp_trade = self.ib.placeOrder(contract, tp_order)
        sl_trade = self.ib.placeOrder(contract, sl_order)

        log.info(f"Bracket exit for seccode={seccode}: TP@{trigger_price_tp}, SL@{trigger_price_sl}, OCA={oca_group}")

        return tp_trade, sl_trade

    def newExitOrder(self, board, seccode, client, exit_action, quantity, trigger_price_sl, trigger_price_tp, correction, spread, bymarket, is_market):
        """ Interactive Brokers """
        try:
            result = self._run_ib(self._newExitOrder_async(seccode, exit_action, quantity, trigger_price_sl, trigger_price_tp))
            if result is None:
                log.error(f"Failed to execute newExitOrder for {seccode} via IB event loop")
                return None, None
            return result

        except Exception as e:
            log.error(f"Error placing bracket exit order for {seccode}: {e}")
            return None, None


    def cancelExitOrder(self, exitOrderId):
        """ Interactive Brokers - thread-safe via _run_ib """
        try:
            self._run_ib(self._cancel_order_async(exitOrderId))
        except Exception as e:
            log.error(f"Error placing cancelExitOrder {exitOrderId}: {e}")
            return None


    def getTradingPlatformTime(self):
        """ Interactive Brokers """
        usEasternTimeZone = pytz.timezone('US/Eastern')
        return datetime.datetime.now(usEasternTimeZone)
    
    
    def cancellAllStopOrders(self):
        pass


    async def _closeExit_async(self, mp, meo_order, sl_meo_order):
        """Run IB operations for closeExit on the IB event loop thread.
        Called via _run_ib() from closeExit() to avoid 'no current event loop' errors
        when invoked from the IB_OrderStatusTask thread.
        """
        # Cancel the TP exit order
        if meo_order is not None:
            self.ib.cancelOrder(meo_order, '')

        # Cancel the SL exit order
        if sl_meo_order is not None:
            self.ib.cancelOrder(sl_meo_order, '')

        # Send market order to close the position
        exit_action = "SELL" if mp.takePosition == "long" else "BUY"
        contract = self._make_stock_contract(mp.seccode)
        expdate = self.convert_to_utc(self.getExpDate(mp.seccode))

        order = MarketOrder(exit_action, mp.quantity)
        order.tif = 'GTD'
        order.goodTillDate = expdate.strftime('%Y%m%d-%H:%M:%S')
        order.account = self.account_number

        trade = self.ib.placeOrder(contract, order)
        return SimpleTrade(trade.order.orderId, trade.orderStatus.status)


    def closeExit(self, mp, meo):
        """ Interactive Brokers """

        try:
            # Find the SL exit order before scheduling async work
            sl_meo = None
            if mp.exit_sl_id is not None:
                sl_meo = next((o for o in self.monitoredExitOrders if o.id == mp.exit_sl_id), None)
                if sl_meo is None:
                    log.warning(f"SL order {mp.exit_sl_id} not found in monitoredExitOrders")

            # Execute IB operations on the IB event loop thread
            sl_meo_order = sl_meo.order if sl_meo is not None else None
            meo_order = meo.order if meo is not None else None
            res = self._run_ib(self._closeExit_async(mp, meo_order, sl_meo_order))

            if res is None:
                log.error(f"Failed to execute closeExit IB operations for {mp.seccode}, keeping position in monitoredPositions")
                mp.exitOrderAlreadyCancelled = False
                return

            # Clean up the SL monitored exit order
            if sl_meo is not None and sl_meo in self.monitoredExitOrders:
                self.monitoredExitOrders.remove(sl_meo)

            if res.status in cm.statusOrderForwarding or res.status in cm.statusOrderExecuted:
                self._market_close_order_ids.add(res.id)
                # Save position so it can be restored if the close order is cancelled
                self._market_close_positions[res.id] = mp
                log.info(f'exit by market successfully processed for {mp.seccode}, close_order_id={res.id}')
            else:
                log.error(f'exit by market failed with status: {res.status} for {mp.seccode}, keeping position in monitoredPositions')
                return

            # Only clean up monitored structures AFTER confirming the market close order was accepted
            if meo is not None:
                if meo in self.monitoredExitOrders:
                    self.monitoredExitOrders.remove(meo)
                self.monitoredPositions = [p for p in self.monitoredPositions if p.exit_tp_id != meo.id]
            else:
                # Position had no exit orders (e.g. reconciled without orders)
                self.monitoredPositions = [p for p in self.monitoredPositions if p.seccode != mp.seccode]

            mp.exitOrderAlreadyCancelled = True

        except Exception as e:
            log.error(f"Error in closeExit {mp} , {meo}: {e}")
            mp.exitOrderAlreadyCancelled = False


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
                order.id in [ m.entry_id, m.exit_tp_id ] 
                or m.exit_order_no == order.id 
                or ( m.seccode == order.seccode and m.buysell == order.buysell  )
            ):
               monitoredPosition = m
               break
        return monitoredPosition

    
    def get_cash_balance(self):
        """ Interactive Brokers """
        if self.MODE == 'TEST_OFFLINE':
            return getattr(cm, 'simulation_net_balance', 29000)
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
            log.debug(f"Cash balance: ${cash_balance}")
            return cash_balance

        except Exception as e:
            log.error(f"Failed to get cash balance: {e}")
            return 0.0


    def get_net_balance(self):
        """ Interactive Brokers """
        if self.MODE == 'TEST_OFFLINE':
            return getattr(cm, 'simulation_net_balance', 29000)
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
            log.debug(f"Net balance (NetLiquidation): ${net_balance}")
            return net_balance

        except Exception as e:
            log.error(f"Error retrieving net balance: {e}")
            return 0.0


    def openPosition(self, position):
        """ Interactive Brokers """

        buysell = "BUY" if position.takePosition == "long" else "SELL"

        # Respect entryByMarket from security config instead of forcing MKT
        sec = next((s for s in self.securities if s['seccode'] == position.seccode), {})
        entry_by_market = sec.get('params', {}).get('entryByMarket', True)
        position.bymarket = entry_by_market

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
            # Track limit entry attempt
            if not entry_by_market:
                position.entry_limit_prices.append({
                    'price': price,
                    'time': datetime.datetime.now(timezone.utc).isoformat(),
                    'order_id': res.id
                })
            self.monitoredPositions.append(position)
            log.info(f"orderId: {res.id}, status: {res.status}, bymarket: {entry_by_market}")

        else:
            log.error(f"Order failed or in invalid state: {res.status}")
        

    def removeMonitoredPositionByExit(self, order):
        """ Interactive Brokers """
        for p in self.monitoredPositions:
            if order.id in (p.exit_tp_id, p.exit_sl_id):
                close_time = datetime.datetime.now(datetime.timezone.utc)
                self.position_closed_at[p.seccode] = close_time
                entry_time = p.entry_time or self.position_entry_filled_at.get(p.seccode)
                if entry_time:
                    duration = (close_time - entry_time).total_seconds()
                    if duration < 60:
                        self.position_scalp_cooldown[p.seccode] = 1800
                        log.info(f"Position closed for {p.seccode} in {duration:.0f}s (scalp detected), cooldown 1800s")
                    else:
                        self.position_scalp_cooldown.pop(p.seccode, None)
                        log.info(f"Position closed for {p.seccode} after {duration:.0f}s, normal cooldown")
                else:
                    log.info(f"Position closed for {p.seccode}, cooldown started")
        self.monitoredPositions = [
            p for p in self.monitoredPositions
            if order.id not in (p.exit_tp_id, p.exit_sl_id)
        ]

    def reconcileOrphanedPositions(self):
        """ Interactive Brokers
        Detects positions whose exit orders (TP and SL) are no longer active in IB
        (e.g. filled or cancelled while bot was offline) and removes them.
        Also detects ghost positions where the entry was rejected by IB (Error 460)
        but the position was already added to monitoredPositions.
        Also repairs positions missing one exit order (TP or SL) by re-creating the bracket.
        IMPORTANT: positions confirmed by IB portfolio are NEVER removed as orphaned;
        instead their exit orders are re-linked from IB open trades.
        """
        active_exit_ids = {o.id for o in self.monitoredExitOrders}
        active_entry_ids = {o.id for o in self.monitoredOrders}
        open_trades = self.ib.openTrades()
        ib_order_ids = {t.order.orderId for t in open_trades}

        # Build set of seccodes with real positions in IB portfolio
        try:
            ib_portfolio_seccodes = {
                p.contract.symbol for p in self.ib.positions()
                if p.position != 0
            }
        except Exception:
            ib_portfolio_seccodes = set()

        orphaned = []
        relink = []   # positions confirmed in IB but with stale/missing exit IDs
        repair = []
        for mp in self.monitoredPositions:
            # Parked positions (market close retry limit reached):
            # - If no longer in IB portfolio → treat as orphaned (auto-remove)
            # - If still in IB portfolio → skip (avoid spam, wait for market open)
            if getattr(mp, 'exitOrderAlreadyCancelled', False) and mp.exit_tp_id is None and mp.exit_sl_id is None:
                if mp.seccode not in ib_portfolio_seccodes:
                    orphaned.append(mp)
                continue
            if mp.exit_tp_id is not None and mp.exit_sl_id is not None:
                tp_active = mp.exit_tp_id in active_exit_ids or mp.exit_tp_id in ib_order_ids
                sl_active = mp.exit_sl_id in active_exit_ids or mp.exit_sl_id in ib_order_ids
                if not tp_active and not sl_active:
                    if mp.seccode in ib_portfolio_seccodes:
                        # IB confirms position exists — stale exit IDs, need re-link
                        relink.append(mp)
                    else:
                        orphaned.append(mp)
                elif tp_active != sl_active:
                    # One exit order is missing — needs bracket repair
                    repair.append(mp)
            elif mp.exit_tp_id is None and mp.exit_sl_id is None:
                # Ghost position: entry was rejected (e.g. Error 460) before being filled.
                # Entry is not in monitoredOrders and not in IB open trades.
                if mp.entry_id not in active_entry_ids and mp.entry_id not in ib_order_ids:
                    if mp.seccode in ib_portfolio_seccodes:
                        # IB confirms position exists — filled entry, exit orders not persisted
                        relink.append(mp)
                    else:
                        orphaned.append(mp)
            else:
                # Mixed case: one exit_id set, the other None — needs bracket repair
                repair.append(mp)

        # Re-link: find correct exit orders from IB open trades
        relinked = False
        for mp in relink:
            exit_action = "SELL" if mp.takePosition == "long" else "BUY"
            exit_orders = [
                t for t in open_trades
                if t.contract.symbol == mp.seccode
                and t.order.action.upper() == exit_action
                and t.order.orderType in ('LMT', 'STP', 'STP LMT')
            ]
            if len(exit_orders) >= 2:
                # Closest to entry = TP, farthest = SL
                exit_orders.sort(key=lambda t: abs(self._get_order_price(t) - mp.entryPrice))
                tp_trade = exit_orders[0]
                sl_trade = exit_orders[1]
                old_tp, old_sl = mp.exit_tp_id, mp.exit_sl_id
                mp.exit_tp_id = tp_trade.order.orderId
                mp.exit_sl_id = sl_trade.order.orderId
                mp.exitOrderRequested = True
                # Register in monitoredExitOrders
                tp_ib = OrderIB(tp_trade)
                sl_ib = OrderIB(sl_trade)
                if tp_ib not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(tp_ib)
                if sl_ib not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(sl_ib)
                log.warning(f"Re-linked exit orders for {mp.seccode}: "
                            f"TP {old_tp}->{mp.exit_tp_id}, SL {old_sl}->{mp.exit_sl_id}")
                relinked = True
            elif len(exit_orders) == 1:
                # Only one exit order found — re-link it and mark for bracket repair
                trade = exit_orders[0]
                if trade.order.orderType == 'LMT':
                    mp.exit_tp_id = trade.order.orderId
                    mp.exit_sl_id = None
                else:
                    mp.exit_tp_id = None
                    mp.exit_sl_id = trade.order.orderId
                mp.exitOrderRequested = True
                ib_order = OrderIB(trade)
                if ib_order not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(ib_order)
                log.warning(f"Re-linked partial exit for {mp.seccode}: "
                            f"TP={mp.exit_tp_id}, SL={mp.exit_sl_id} — needs bracket repair")
                # Refresh sets for repair
                active_exit_ids = {o.id for o in self.monitoredExitOrders}
                repair.append(mp)
                relinked = True
            else:
                # No exit orders in IB — position exists but unprotected, create bracket
                log.warning(f"Position {mp.seccode} confirmed in IB portfolio but has NO exit orders "
                            f"— will attempt bracket repair")
                mp.exit_tp_id = None
                mp.exit_sl_id = None
                repair.append(mp)
                relinked = True

        # Persist re-linked state to DB
        if relinked:
            self.storeMonitoredPositions()

        for mp in orphaned:
            close_time = datetime.datetime.now(datetime.timezone.utc)
            entry_time = mp.entry_time or self.position_entry_filled_at.get(mp.seccode)
            if entry_time and (close_time - entry_time).total_seconds() < 60:
                self.position_scalp_cooldown[mp.seccode] = 1800
            log.warning(f"Orphaned position detected: {mp.seccode} "
                        f"(exit orders TP={mp.exit_tp_id} SL={mp.exit_sl_id} no longer active) "
                        f"- removing from monitoredPositions")
            self.position_closed_at[mp.seccode] = close_time
            self.monitoredPositions.remove(mp)
        for mp in repair:
            self._repairExitBracket(mp, active_exit_ids, ib_order_ids)

    def _repairExitBracket(self, mp, active_exit_ids, ib_order_ids):
        """Cancel the surviving exit order and re-create a full TP+SL bracket."""
        # Only attempt repair once per position to avoid infinite loops
        if not hasattr(self, '_bracket_repair_attempted'):
            self._bracket_repair_attempted = set()
        if mp.seccode in self._bracket_repair_attempted:
            return
        self._bracket_repair_attempted.add(mp.seccode)

        tp_active = (mp.exit_tp_id in active_exit_ids or mp.exit_tp_id in ib_order_ids) if mp.exit_tp_id is not None else False
        sl_active = (mp.exit_sl_id in active_exit_ids or mp.exit_sl_id in ib_order_ids) if mp.exit_sl_id is not None else False

        if not tp_active and not sl_active:
            missing = "TP+SL"
        elif tp_active and not sl_active:
            missing = "SL"
        else:
            missing = "TP"
        surviving_id = mp.exit_tp_id if tp_active else (mp.exit_sl_id if sl_active else None)

        log.warning(f"Repairing bracket for {mp.seccode}: {missing} order missing "
                    f"(TP={mp.exit_tp_id} active={tp_active}, SL={mp.exit_sl_id} active={sl_active})")
        try:
            # Cancel the surviving order (if any)
            if surviving_id is not None:
                self.cancel_order(surviving_id)
                self.monitoredExitOrders = [o for o in self.monitoredExitOrders if o.id != surviving_id]
                log.info(f"Cancelled surviving {('TP' if tp_active else 'SL')} order {surviving_id} for {mp.seccode}")

            # Re-create full bracket
            exit_action = "SELL" if mp.takePosition == "long" else "BUY"
            trigger_price_tp = "{0:0.{prec}f}".format(
                round(mp.exitPrice, mp.decimals), prec=mp.decimals)
            trigger_price_sl = "{0:0.{prec}f}".format(
                round(mp.stoploss, mp.decimals), prec=mp.decimals)

            res_tp, res_sl = self.newExitOrder(
                None, mp.seccode, None, exit_action,
                mp.quantity, trigger_price_sl, trigger_price_tp,
                None, None, None, False)

            if res_tp is not None and res_sl is not None:
                mp.exit_tp_id = res_tp.order.orderId
                mp.exit_sl_id = res_sl.order.orderId
                mp.exitOrderRequested = True
                # Register in monitoredExitOrders
                tp_order_ib = OrderIB(res_tp)
                sl_order_ib = OrderIB(res_sl)
                if tp_order_ib not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(tp_order_ib)
                if sl_order_ib not in self.monitoredExitOrders:
                    self.monitoredExitOrders.append(sl_order_ib)
                log.info(f"Bracket repaired for {mp.seccode}: new TP={mp.exit_tp_id}, new SL={mp.exit_sl_id}")
                self.storeMonitoredPositions()
            else:
                log.error(f"Failed to repair bracket for {mp.seccode}: newExitOrder returned None")
        except Exception as e:
            log.error(f"Error repairing bracket for {mp.seccode}: {e}")

    def reconcileBrokerPositions(self):
        """Interactive Brokers
        Detects positions open in IB that are NOT in monitoredPositions
        (e.g. due to crash, manual intervention) and recovers them so they
        get proper TP/SL management and timeout handling.
        Called once at startup after monitoredPositions is loaded from DB.
        """
        log.warning("RECONCILE: Starting broker position reconciliation...")
        try:
            # 1. Get real holdings from IB
            ib_positions = self.ib.positions()
            # Filter by our account
            ib_positions = [p for p in ib_positions if p.account == self.account_number]

            if not ib_positions:
                log.warning("RECONCILE: No positions found in IB portfolio")
                return

            # 2. Build set of seccodes already monitored
            monitored_seccodes = {mp.seccode for mp in self.monitoredPositions}

            # 3. Request ALL open orders (from all clientIds) so we can match
            #    exit orders that may have been placed by a previous session
            try:
                async def _req_all_open_orders():
                    return await self.ib.reqAllOpenOrdersAsync()
                self._run_ib(_req_all_open_orders(), timeout=10)
            except Exception as e:
                log.warning(f"RECONCILE: reqAllOpenOrders failed: {e}, proceeding with cached trades")
            open_trades = self.ib.openTrades()

            recovered = 0
            for item in ib_positions:
                seccode = item.contract.symbol
                if seccode in monitored_seccodes:
                    continue
                if item.position == 0:
                    continue

                # Determine direction
                direction = "long" if item.position > 0 else "short"
                quantity = abs(item.position)
                entry_price = item.avgCost

                # Find security config
                sec = next((s for s in self.securities if s['seccode'] == seccode), None)
                if sec is None:
                    log.warning(f"RECONCILE: {seccode} found in IB but not in securities config, skipping")
                    continue

                board = sec.get('board', 'EQTY')
                market_id = sec.get('id', 0)
                decimals = sec.get('decimals', 2)

                # Find open exit orders for this seccode
                exit_action = "SELL" if direction == "long" else "BUY"
                exit_orders = []
                for trade in open_trades:
                    if (trade.contract.symbol == seccode
                            and trade.order.action.upper() == exit_action
                            and trade.order.orderType in ('LMT', 'STP', 'STP LMT')):
                        exit_orders.append(trade)

                exit_tp_id = None
                exit_sl_id = None
                exit_price = None
                stoploss = None
                tp_trade = None
                sl_trade = None

                if len(exit_orders) >= 2:
                    # Sort by distance to entry_price; closest = TP, farthest = SL
                    exit_orders.sort(key=lambda t: abs(self._get_order_price(t) - entry_price))
                    tp_trade = exit_orders[0]
                    sl_trade = exit_orders[1]
                    exit_tp_id = tp_trade.order.orderId
                    exit_sl_id = sl_trade.order.orderId
                    exit_price = self._get_order_price(tp_trade)
                    stoploss = self._get_order_price(sl_trade)
                elif len(exit_orders) == 1:
                    tp_trade = exit_orders[0]
                    exit_tp_id = tp_trade.order.orderId
                    exit_price = self._get_order_price(tp_trade)
                    stoploss = entry_price  # placeholder, no SL order found
                else:
                    # No exit orders at all
                    exit_price = entry_price
                    stoploss = entry_price

                # Create the Position object
                pos = Position(
                    takePosition=direction,
                    board=board,
                    seccode=seccode,
                    marketId=market_id,
                    quantity=quantity,
                    entryPrice=entry_price,
                    exitPrice=exit_price,
                    stoploss=stoploss,
                    decimals=decimals,
                    client=self.getClientId(),
                    entry_id=None,
                    exit_tp_id=exit_tp_id,
                    exit_sl_id=exit_sl_id,
                    entry_time=datetime.datetime.now(datetime.timezone.utc),
                )

                self.monitoredPositions.append(pos)
                monitored_seccodes.add(seccode)

                # Register exit orders in monitoredExitOrders so reconcileOrphanedPositions
                # does not immediately remove them
                if tp_trade is not None:
                    self.monitoredExitOrders.append(OrderIB(tp_trade))
                if sl_trade is not None:
                    self.monitoredExitOrders.append(OrderIB(sl_trade))

                log.warning(
                    f"RECONCILE: Recovered {direction} position for {seccode}: "
                    f"qty={quantity}, entry={entry_price}, "
                    f"TP={exit_price} (id={exit_tp_id}), SL={stoploss} (id={exit_sl_id})"
                )
                recovered += 1

            if recovered > 0:
                self.storeMonitoredPositions()
                log.warning(f"RECONCILE: Recovered {recovered} position(s) from IB broker, persisted to DB")
            else:
                log.warning("RECONCILE: All IB portfolio positions match monitoredPositions, no recovery needed")

        except Exception as e:
            log.error(f"RECONCILE: Error during broker position reconciliation: {e}", exc_info=True)

    @staticmethod
    def _get_order_price(trade):
        """Extract the effective price from an IB Trade object (LMT or STP)."""
        order = trade.order
        if order.orderType == 'LMT' and order.lmtPrice:
            return order.lmtPrice
        if order.orderType in ('STP', 'STP LMT') and order.auxPrice:
            return order.auxPrice
        return 0.0

    def set_exit_order_no_to_MonitoredPosition (self, stopOrder):
        """ Interactive Brokers """
        pass

    
    def triggerExitOrder(self, order, monitoredPosition):
        """ Interactive Brokers """    
        
        log.info('Triggering stop order...')        
        # Determine buy/sell action based on position type
        exit_action = "SELL" if monitoredPosition.takePosition == "long" else "BUY"
        # Format trigger prices
        trigger_price_tp = "{0:0.{prec}f}".format(
            round(monitoredPosition.exitPrice, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        trigger_price_sl = "{0:0.{prec}f}".format(
            round(monitoredPosition.stoploss, monitoredPosition.decimals), prec=monitoredPosition.decimals
        )
        
        res, res_sl = self.newExitOrder(
            None, monitoredPosition.seccode, None, exit_action, 
            monitoredPosition.quantity, trigger_price_sl, trigger_price_tp,
            None, None, None, False 
        )
        
        if res is None:
            log.error("Failed to create Exit order: newExitOrder returned None")
            
        elif res.orderStatus.status in cm.statusOrderForwarding or res.orderStatus.status in cm.statusOrderExecuted:

            monitoredPosition.exitOrderRequested = True
            monitoredPosition.exit_tp_id = res.order.orderId  # Capture IB order ID
            monitoredPosition.exit_sl_id = res_sl.order.orderId
            log.info(f"Exit order {order.id} successfully in IB OrderId: {res.order.orderId}")
            log.info(repr(res))

            # Immediately register exit orders in monitoredExitOrders to prevent
            # reconcileOrphanedPositions from removing this position before
            # processExitOrderStatus has a chance to add them
            tp_order_ib = OrderIB(res)
            sl_order_ib = OrderIB(res_sl)
            if tp_order_ib not in self.monitoredExitOrders:
                self.monitoredExitOrders.append(tp_order_ib)
            if sl_order_ib not in self.monitoredExitOrders:
                self.monitoredExitOrders.append(sl_order_ib)

            # Persist exit order IDs to DB so they survive restart
            self.storeMonitoredPositions()

            # safety net to not collision with an Entry
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)               
            
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
            contract = self._make_stock_contract(seccode)

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
     
