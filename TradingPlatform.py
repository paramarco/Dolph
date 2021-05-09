# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:11:34 2020

@author: mvereda
"""
import logging
import time
import datetime
import pytz
import copy
import sys
from threading import Thread

import lib.transaq_connector.commands as tc
import lib.transaq_connector.structures as ts

log = logging.getLogger("TradingPlatform")

class Position:
    def __init__(self, takePosition, board, seccode, marketId, entryTimeSeconds,
                 quantity, entryPrice, exitPrice, stoploss, 
                 decimals, exitTime, correction, spread, bymarket = False ):
        
        # id:= transactionid of the first order, "your entry" of the Position
        # will be assigned once upon successful entry of the Position
        self.entry_id = None
        # will be assigned once upon successful entry of the Position
        self.exit_id = None
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
        self.stopOrderRequested = False
        # to be assigned when position is being proccessed by Tradingplatform
        self.buysell = None
        # exitTime := time for a emergency exit, close current position at 
        # this time by market if the planned exit is not executed yet
        self.exitTime = exitTime
        
        self.correction = correction
        self.spread = spread               
        self.bymarket = bymarket
        
        self.client = None
        self.union = None
        self.expdate = None

    def __str__(self):
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
        fmt = "%d.%m.%Y %H:%M:%S"
        msg += ' exitTime=' + self.exitTime.strftime(fmt)
        

        
        return msg
    
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
                    res = self.tc.get_history(
                        s['board'], 
                        s['seccode'], 
                        1, 
                        2, 
                        True
                    )
                    log.debug(repr(res))
                else:
                    log.warning('wait!, not connected to TRANSAQ yet...')            
            time.sleep(1)    

class TradingPlatform:
    def __init__(self, target, securities, onHistoryCandleCall, 
                 connectOnInit, onCounterPosition):
        
        log.info('TradingPlatform starting  ...')
        
        self.securities = securities
        self.onHistoryCandleCall = onHistoryCandleCall
        self.onCounterPosition = onCounterPosition
        self.clientAccounts = []
        self.monitoredPositions = []
        self.monitoredOrders = []
        self.monitoredStopOrders = []
        self.counterPositions = []
        
        self.tc = tc.TransaqConnector()
        self.tc.connected = False
        self.profitBalance = 0
        self.currentTradingHour = 0
        
        self.candlesUpdateThread = None
        self.candlesUpdateTask = None
        
        self.fmt = "%d.%m.%Y %H:%M:%S"
        
        if connectOnInit:
            self.tc.initialize("log", 3, self.handle_txml_message)   
            self.connect2TRANSAQ()        
    
    def handle_txml_message(self, obj):

        if isinstance(obj, ts.CmdResult):
            logging.info( repr(obj) )
        elif isinstance(obj, ts.ClientOrderPacket):
            self.onClientOrderPacketRes(obj)
        elif isinstance(obj, ts.ClientAccount):
            self.onClientAccountRes(obj)
        elif isinstance(obj, ts.ServerStatus):
            if obj.connected == 'true':
                self.setConnected2Transaq()
            else:
                self.disconnect()
                self.tc.connected = False
                while self.tc.connected == False:
                    log.info('waiting 60 seconds to establish a connection...')
                    time.sleep(60)
                    self.tc.initialize("log", 3, self.handle_txml_message) 
                    self.connect2TRANSAQ()
                    
        elif isinstance(obj, ts.HistoryCandlePacket):
            self.onHistoryCandleCall(obj)
            self.cancelTimedoutEntries()
            self.cancelTimedoutExits()
            self.cancelHangingOrders()
            self.updateTradingHour()
            
        elif isinstance(obj, ts.MarketPacket):
            pass # logging.info( repr(obj) ) 
            
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

 
    
    def connect2TRANSAQ(self):
        log.info('connecting to TRANSAQ...')
        # this is my saparte transaq conto for dolph
        res = self.tc.connect("FZTC14861A", "Lovemoney2018", "tr1.finam.ru:3900")    
        # res = self.tc.connect("FZTC8927A", "Lovemoney2018", "tr1.finam.ru:3900")    
        # res = self.tc.connect("FZTC8929A", "vereda", "tr1.finam.ru:3900")
        

        log.debug(repr(res))
   
    def disconnect(self):
        
        log.info('stopping candlesUpdateTask ...')
        self.candlesUpdateTask.terminate()
        
        if self.tc.connected:
            log.info('disconnecting from TRANSAQ...')
            self.tc.disconnect()
            self.tc.uninitialize()
        
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
        
        
    def get_history(self, board, seccode, period, count, reset=True):
        if self.tc.connected == True:
            res = self.tc.get_history(board, seccode, period, count, reset=True)
            log.debug(repr(res))
        else:
            log.warning('wait!, not connected to TRANSAQ yet...')            
   
    def HistoryCandleReq (self, securities, period, count = 2):
        for s in securities:
            self.get_history(s['board'], s['seccode'], 1 , count) 
            
        time.sleep(2)        
        
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
        
    def triggerStopOrder(self, order, monitoredPosition):
        
        buysell = ""
        if      monitoredPosition.buysell == "B":   buysell = "S";
        elif    monitoredPosition.buysell == "S":   buysell = "B";
        else:   raise Exception("what? " + monitoredPosition.buysell);
        
        trigger_price_tp = "{0:0.{prec}f}".format(
            round(monitoredPosition.exitPrice, monitoredPosition.decimals),
            prec = monitoredPosition.decimals
        )
        trigger_price_sl = "{0:0.{prec}f}".format(
            round(monitoredPosition.stoploss, monitoredPosition.decimals),
            prec = monitoredPosition.decimals
        )                 
        res = self.tc.new_stoporder(
            order.board, order.seccode, order.client, buysell, 
            monitoredPosition.quantity,trigger_price_sl,trigger_price_tp,
            monitoredPosition.correction, monitoredPosition.spread, 
            monitoredPosition.bymarket, False 
        )    
        log.info(repr(res))
        if res.success :
            monitoredPosition.stopOrderRequested = True;
            monitoredPosition.exit_id = res.id                
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
            
            m="stopOrder of order {} succesfully requested".format(order.id)
            m+=", deleted from monitored Orders"""
            logging.info( m )                
        else:
            monitoredPosition.stopOrderRequested = False
            logging.error("takeprofit hasn't been processed by transaq")      
        
    def triggerWhenMatched(self, order):
        
        trigger = False
        transactionId = None
        position = None
        
        for cp in self.counterPositions:
            transactionId, position = cp
            if order.id == transactionId:
                trigger = True
                break

        
        if trigger == True:
            m = "trigering onCounterPosition for {}".format( str(position) )
            logging.info( m )
            position2invert = copy.deepcopy(position)
            self.counterPositions = \
            list(filter(lambda x: x[0] != transactionId, self.counterPositions))            
            
            self.onCounterPosition(position2invert)
            
            
            
        
    def processOrderStatus(self, order):
        
        logging.info( repr(order) )        
        s = order.status
        monitoredPosition = None

        if s == 'matched':
            
            clone = copy.deepcopy(order)            
            for m in self.monitoredPositions:
                if m.entry_id == order.id: monitoredPosition = m; break;
                if m.exit_id == order.id: monitoredPosition = m; break;
                    
            if monitoredPosition is None :
                m ='already processed, deleting: {}'.format( repr(order))
                logging.info( m )
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order) 
                
            elif monitoredPosition.stopOrderRequested == False :
                self.triggerStopOrder(order, monitoredPosition)                
            else:                
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order)
                m = "{} already requested before, deleted".format(order.id)                
                logging.info( m ) 
                
            self.triggerWhenMatched(clone)            
            
        elif s in ['watching','active','forwarding']:
            
            if order not in self.monitoredOrders:
                if order.time is None:
                    order.time = self.getMoscowTime()
                self.monitoredOrders.append(order)
                m = 'order {} with status: {} added to monitoredOrders'.format( order.id, s)
                logging.info( m )                
           
        elif s in ['rejected','expired','denied','cancelled','removed'] :
            
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
                self.tc.cancel_order(order.id)                
            self.monitoredPositions = [ p for p in self.monitoredPositions 
                                         if p.entry_id != order.id ]

            m = 'order {} with status: {} deleted from monitoredOrders'.format( order.id, s)
            logging.info( m )
                    
        else :            
            others = '"none","inactive","wait","disabled","failed","refused"'
            m = 'status: {} , belongs to: {}'.format( s, others)
            logging.info( m )
            
        self.reportCurrentOpenPositions()

    def processStopOrderStatus(self, stopOrder):
        
        logging.info( repr(stopOrder) )
        
        s = stopOrder.status
        
        if s in ['tp_guardtime','tp_forwarding','watching',
                 "sl_forwarding","sl_guardtime"] :
            
            if stopOrder not in self.monitoredStopOrders:
                self.monitoredStopOrders.append(stopOrder)
                m = 'stopOrder {} with status: {} added to monitoredStopOrders'
                m = m.format( stopOrder.id, s)
                logging.info( m )                


        elif s in ['tp_executed','sl_executed','cancelled','denied','disabled',
                   'expired','failed','rejected']:
            if stopOrder in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(stopOrder)
            
                m='id: {} with status: {} deleted from monitoredStopOrders'
                m = m.format( stopOrder.id, s )
                logging.info( m )
                self.monitoredPositions = [ p for p in self.monitoredPositions 
                                             if p.exit_id != stopOrder.id ] 
                self.updatePortfolioPerformance(s)
           
        else:            
            others = '"linkwait","tp_correction","tp_correction_guardtime"'
            m = 'status: {} skipped, belongs to: {}'.format( s, others)
            logging.info( m )
        
        self.reportCurrentOpenPositions()
        
    def isPositionOpen( self, seccode ):
        """ 
            isExitOrderNotTriggered is True when there is a position which 
            takeProfit/stoploss was not yet triggered.
            
            isExitOrderActive is True when there isn't any MonitoredPosition
            and the exit order is still active hanging, not executed yet.
            
            isPositionOpen = isExitOrderNotTriggered or isExitOrderActive                
        """
        flag = False
        inMP = False
        inMSP = False
        isOrderActive = False

        for mp in self.monitoredPositions:
           if mp.seccode == seccode:
               inMP = True
               break
        
        for msp in self.monitoredStopOrders:
           if msp.seccode == seccode:
               inMSP = True
               break
              
        for mo in self.monitoredOrders:
            if mo.seccode == seccode:
                isOrderActive = True
                break
            
        isExitOrderNotTriggered = inMP and ( inMSP or isOrderActive )
        
        isExitOrderActive = not inMP and isOrderActive
      
        flag = isExitOrderNotTriggered or isExitOrderActive 
        
        return flag
             
            
    def reportCurrentOpenPositions(self):
        
        numMonPosition = len(self.monitoredPositions)
        numMonOrder = len(self.monitoredOrders)
        numMonStopOrder = len(self.monitoredStopOrders)
        
        msg = "\n"
        msg += 'monitored Positions: '+str(numMonPosition)+'\n'
        for mp in self.monitoredPositions:
            msg += str(mp) + '\n'
            
        msg += 'monitored Orders   (entries): '+str(numMonOrder)+'\n'
        for mo in self.monitoredOrders:
            msg += repr(mo) + '\n'
            
        msg += 'monitored StopOrders (exits): '+str(numMonStopOrder)+'\n'
        for mso in self.monitoredStopOrders:
            msg +=  repr(mso) + '\n'
        
        logging.info( msg )        
        total = numMonOrder + numMonStopOrder        

        return total
    
    def getExpDate(self, seccode):

        nSec = 0
        for sec in self.securities :
            if seccode == sec['seccode']:
                nSec = sec['params']['ActiveTimeSeconds']
        
        if nSec == 0 : log.error('this shouldnt happen'); return;                
                
        moscowTime = self.getMoscowTime()
        moscowTime_plusNsec = moscowTime + datetime.timedelta(seconds = nSec)
        expdate = moscowTime_plusNsec.strftime(self.fmt) 
        
        
        return expdate

        
    
    def openPosition ( self, position):        

        position.expdate = self.getExpDate(position.seccode)
        price = round(position.entryPrice , position.decimals)
        price = "{0:0.{prec}f}".format(price,prec=position.decimals)        
        
        res = self.tc.new_order(
            position.board,position.seccode,position.client,position.union,
            position.buysell, position.expdate, position.union, 
            position.quantity,price, position.bymarket,False
        )
        log.debug(repr(res))
        if res.success == True:
            position.entry_id = res.id
            self.monitoredPositions.append(position)                                
        else:
            logging.error( "position has not been processed by transaq")
       
    def getMonitoredPositionBySeccode(self, seccode):
        
        monitoredPosition = None
        for mp in self.monitoredPositions:
            if mp.seccode == seccode:
                monitoredPosition = mp
                break
        if monitoredPosition is None:
            logging.error( "monitoredPosition Not found, recheck this case")
            
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
    
    def closePosition( self, position, withCounterPosition = False):
        
        code = position.seccode        
        monitoredPosition = self.getMonitoredPositionBySeccode(code)
        monitoredStopOrder = self.getMonitoredStopOrderBySeccode(code)        
        
        if monitoredPosition is None or monitoredStopOrder is None:
            logging.error("position Not found, recheck this case")
        else:
            log.info('close action received, closing position...')
            cloneMP = copy.deepcopy(monitoredPosition)
            tid = self.closeExit( monitoredPosition, monitoredStopOrder )
            if withCounterPosition == True:
                log.info('adding position to counterPositions ...') 
                cloneMP.stopOrderRequested = False
                cloneMP.entry_id = None
                cloneMP.exit_id = None
                cp = ( tid, cloneMP )
                self.counterPositions.append(cp)            
   

    def processPosition ( self, position):
        
        self.reportCurrentOpenPositions()        
        if self.tc.connected == False:
            log.warning('wait!, not connected to TRANSAQ yet...')
            return
        if self.isPositionOpen(position.seccode) and \
           position.takePosition not in ['close','close-counterPosition']:
            msg='there is a position opened for {}'.format(position.seccode)            
            logging.warning( msg )
            return        

        position.client = self.getClientIdByMarket(position.marketId)
        position.union =  self.getUnionIdByMarket(position.marketId)        
        
        if (position.takePosition == "long"):           
            position.buysell = "B"
            self.openPosition( position )
        elif (position.takePosition  == "short"): 
            position.buysell = "S"
            self.openPosition( position )
        elif position.takePosition == "close":            
            self.closePosition( position )
        elif position.takePosition == "close-counterPosition":
            self.closePosition( position, withCounterPosition = True )
        else:
            logging.error( "takePosition must be either long,short or close")
            raise Exception( position.takePosition )
        
        self.reportCurrentOpenPositions()        
        
        
        
    def cancellAllStopOrders(self):
        if self.tc.connected == True:
            for mso in self.monitoredStopOrders:
                res = self.tc.cancel_takeprofit(mso.id)
                log.debug(repr(res))
            log.debug('finished!')
        else:
            log.warning('wait!, not connected to TRANSAQ yet...')  

    def cancellAllOrders(self):
        if self.tc.connected == True:
            for mo in self.monitoredOrders:
                res = self.tc.cancel_order(mo.id)
                log.debug(repr(res))
            log.debug('finished!')
        else:
            log.warning('wait!, not connected to TRANSAQ yet...')  
        
    def cancelTimedoutEntries(self):
        
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = datetime.datetime.now(moscowTimeZone) 
        list2cancel = []
        
        for mp in self.monitoredPositions:
            for mo in self.monitoredOrders:
                if mp.entry_id == mo.id:
                    nSec = mp.entryTimeSeconds
                    orderTime_plusNsec = mo.time + datetime.timedelta(seconds = nSec)
                    orderTime_plusNsec = moscowTimeZone.localize(orderTime_plusNsec)
                    if ( moscowTime > orderTime_plusNsec ):
                        res = self.tc.cancel_order(mo.id)
                        log.debug(repr(res))
                        list2cancel.append(mo)
                        moscow = moscowTime.strftime(self.fmt)
                        expTime = orderTime_plusNsec.strftime(self.fmt)
                        msg = 'moscow: '+ moscow + ' entry timedouts at: '
                        msg += expTime + ' '+ repr(mo)
                        log.info(msg)
                        
                    break
                   
                    
        for mo in list2cancel:
            if mo in self.monitoredOrders:
                self.monitoredOrders.remove(mo)
                self.monitoredPositions = [ p for p in self.monitoredPositions 
                                             if p.entry_id != mo.id ]

    def cancelTimedoutExits(self):
        
        moscowTime = self.getMoscowTime()        
        
        for mp in self.monitoredPositions:
            for mso in self.monitoredStopOrders:
                if mp.exit_id == mso.id:
                    if ( moscowTime > mp.exitTime ):
                        log.info( 'time-out exit detected, closing exit' )
                        self.closeExit(mp,mso)
                    break
                   
        
    def updatePortfolioPerformance(self, status):
        
        if status == 'tp_executed':
            self.profitBalance += 1
                
        elif status == 'sl_executed':
            self.profitBalance -= 1
        else:
            m='status: {} does not update the portfolio performance'.format(status)
            logging.info( m )
        
        logging.info('portforlio balance: {}'.format(self.profitBalance))       

            
        
    def updateTradingHour(self):
        
        moscowTime = self.getMoscowTime()
        currentHour = moscowTime.hour
        
        if self.currentTradingHour != currentHour :
            self.currentTradingHour = currentHour
            self.profitBalance = 0
            logging.debug('hour changed ... profitBalance has been reset ')
            
    def getProfitBalance(self):
        return self.profitBalance
    
    def getMoscowTime(self):
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = datetime.datetime.now(moscowTimeZone)
        
        return moscowTime
    
    def closeExit(self, mp, mso):       
        
        moscowTime = self.getMoscowTime()        
        list2cancel = []
        tid = None

        res = self.tc.cancel_stoploss(mso.id)
        log.debug(repr(res))
        if res.success == True:
            list2cancel.append(mso)
            moscow = moscowTime.strftime(self.fmt)
            exitTime = mp.exitTime.strftime(self.fmt)
            msg = 'moscow: '+moscow+' exit timedouts at: '+ exitTime
            msg += ' ' + repr(mso)
            log.info( msg )
            res = self.tc.new_order(
                mp.board,
                mp.seccode,
                mp.client,
                mp.union,
                mso.buysell,
                mp.expdate,
                mp.union,
                mp.quantity,
                price=0,
                bymarket = True,
                usecredit = False
            )
            log.debug(repr(res))
            if res.success == True:
                log.info( ' exit request was successfuly processed' )
                tid = res.id
            else:
                log.error( 'exit request was erroneously processed' )
            
        else:
            logging.error( "cancel stop-order error by transaq")


        for mso in list2cancel:
            if mso in self.monitoredStopOrders:
                self.monitoredStopOrders.remove(mso)
            self.monitoredPositions = [ p for p in self.monitoredPositions 
                                             if p.exit_id != mso.id ] 
        
        return tid
        
    def cancelHangingOrders(self):        
        
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = datetime.datetime.now(moscowTimeZone)   
        list2cancel = []      
        
        for mo in self.monitoredOrders:
            nSec = 0
            for sec in self.securities :
                if mo.seccode == sec['seccode']:
                    nSec = sec['params']['ActiveTimeSeconds']
            
            if nSec == 0 : log.error('this shouldnt happen'); return;
                    
            orderTime_plusNsec = mo.time + datetime.timedelta(seconds = nSec)
            orderTime_plusNsec = moscowTimeZone.localize(orderTime_plusNsec)
            if ( moscowTime > orderTime_plusNsec ):
                list2cancel.append(mo)
                msg = 'Order hanging since: ' + mo.time.strftime(self.fmt)
                log.info(msg)                
                   
                    
        for mo in list2cancel:
            if mo in self.monitoredOrders:
                clone = copy.deepcopy(mo)
                self.monitoredOrders.remove(mo)
                res = self.tc.cancel_order(clone.id)
                if res.success == True:
                    res = self.tc.new_order(
                        clone.board,
                        clone.seccode,
                        clone.client,
                        clone.union,
                        clone.buysell,
                        self.getExpDate(clone.seccode),
                        clone.union,
                        clone.quantity,
                        price=0,
                        bymarket = True,
                        usecredit = False
                    )
                    if res.success == True:
                        log.info('exit was successfuly processed'+repr(clone))
                    else:
                        log.error('exit was erroneously processed')
                else:
                    log.error("cancel active-order error by transaq")   
        
        
        
        
