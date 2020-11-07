# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:11:34 2020

@author: mvereda
"""
import logging
import time
import datetime
import pytz

import lib.transaq_connector.commands as tc
import lib.transaq_connector.structures as ts

log = logging.getLogger("TradingPlatform")

class Position:
    def __init__(self, takePosition, board, seccode, marketId, entryTimeSeconds,
                 quantity, entryPrice, exitPrice, stoploss, 
                 decimals, bymarket = False ):
        
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
        # entryTimeSeconds := cancel position if it isn't executed within this seconds 
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
        self.bymarket = bymarket

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
        

        
        return msg
                

class TradingPlatform:
    def __init__(self, target, securities, onHistoryCandleCall, connectOnInit):
        
        log.info('TradingPlatform starting  ...')
        
        self.securities = securities
        self.onHistoryCandleCall = onHistoryCandleCall
        self.clientAccounts = []
        self.monitoredPositions = []
        self.monitoredOrders = []
        self.monitoredStopOrders = []
        self.numMaxOpenPositions = 1
        
        self.tc = tc.TransaqConnector()
        self.tc.connected = False        

        
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
                logging.info('connected to TRANSAQ' )
        elif isinstance(obj, ts.HistoryCandlePacket):
            self.onHistoryCandleCall(obj)
        elif isinstance(obj, ts.MarketPacket):
            logging.info( repr(obj) ) 
            
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
        res = self.tc.connect("FZTC8927A", "Lovemoney2018", "tr1.finam.ru:3900")        
        log.debug(repr(res))
   
    def disconnect(self):
        if self.tc.connected:
            log.info('disconnecting from TRANSAQ...')
            self.tc.disconnect()
            self.tc.uninitialize()
        
    def setConnected2Transaq(self):
        self.tc.connected = True
        log.info('requesting last 300 entries of the securities ...')
        self.HistoryCandleReq( self.securities, 1, 300)
        self.cancellAllOrders()
        self.cancellAllStopOrders()
        
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
        
    def processOrderStatus(self, order):
        
        logging.debug( repr(order) )
        
        s = order.status
        monitoredPosition = None

        if s == 'matched':
            
            for m in self.monitoredPositions:
                if m.entry_id == order.id: monitoredPosition = m; break;
                     
            if monitoredPosition is None :
                m ='already processed, deleting: {}'.format( repr(order))
                logging.debug( m )
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order)                
                                    
            elif monitoredPosition.stopOrderRequested == False :
                
                board = order.board;    seccode = order.seccode;
                client = order.client;  buysell = ""
                if      monitoredPosition.buysell == "B":   buysell = "S";
                elif    monitoredPosition.buysell == "S":   buysell = "B";
                else:   raise Exception("what? " + monitoredPosition.buysell);
                quantity = monitoredPosition.quantity                
                trigger_price_tp = "{0:0.{prec}f}".format(
                    round(monitoredPosition.exitPrice, monitoredPosition.decimals),
                    prec = monitoredPosition.decimals
                )
                trigger_price_sl = "{0:0.{prec}f}".format(
                    round(monitoredPosition.stoploss, monitoredPosition.decimals),
                    prec = monitoredPosition.decimals
                ) 
                correction = 0;    bymarket = monitoredPosition.bymarket; 
                usecredit = False;    
                res = self.tc.new_stoporder(
                    board, seccode, client, buysell, quantity, trigger_price_sl,
                    trigger_price_tp, correction, bymarket, usecredit 
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
            else:
                
                if order in self.monitoredOrders:
                    self.monitoredOrders.remove(order)
                m="stopOrder of order {} already requested before, ".format(order.id)
                m+="deleted from monitored Orders"
                logging.info( m )                
            
        elif s in ['watching','active','forwarding']:
            
            if order not in self.monitoredOrders:
                self.monitoredOrders.append(order)
                m = 'order {} with status: {} added to monitoredOrders'.format( order.id, s)
                logging.info( m )                
           
        elif s in ['rejected','expired','denied','cancelled'] :
            
            if order in self.monitoredOrders:
                self.monitoredOrders.remove(order)
            m = 'order {} with status: {} deleted from monitoredOrders'.format( order.id, s)
            logging.info( m )
                    
        else :            
            others = '"none","inactive","wait","disabled","failed","refused"'
            others += ',"removed" '
            m = 'status: {} , belongs to: {}'.format( s, others)
            logging.debug( m )
            
        self.reportCurrentOpenPositions()

    def processStopOrderStatus(self, stopOrder):
        
        logging.debug( repr(stopOrder) )
        
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
            
                m='takeProfit {} with status: {} deleted from monitoredStopOrders'
                m = m.format( stopOrder.id, s )
                logging.info( m )
                self.monitoredPositions = [ p for p in self.monitoredPositions 
                                             if p.exit_id != stopOrder.id ] 
           
        else:            
            others = '"linkwait","tp_correction","tp_correction_guardtime"'
            m = 'status: {} skipped, belongs to: {}'.format( s, others)
            logging.debug( m )
        
        self.reportCurrentOpenPositions()
            
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
        
        # total = numMonPosition + numMonOrder + numMonStopOrder     
        # total = numMonPosition 

        total = numMonOrder + numMonStopOrder        

        return total

    def processPosition ( self, position):
        
        if self.tc.connected == False:
            log.warning('wait!, not connected to TRANSAQ yet...')
            return
        
        numCurrentOpenPosition = self.reportCurrentOpenPositions()
        
        if numCurrentOpenPosition >= self.numMaxOpenPositions:
            msg='limit of open positions exceed: {} of {}'.format(
                numCurrentOpenPosition, self.numMaxOpenPositions 
            )            
            logging.info( msg )
            return
        

        board = position.board
        seccode = position.seccode
        client = self.getClientIdByMarket(position.marketId)
        union =  self.getUnionIdByMarket(position.marketId)        
        buysell = ""
        
        if (position.takePosition == "long"):
            if any( mso.buysell != 'S' for mso in self.monitoredStopOrders):
                logging.error( "there is a Position still open for short")
                return            
            buysell = "B"
        elif (position.takePosition  == "short"):
            if any( mso.buysell != 'B' for mso in self.monitoredStopOrders):
                logging.error( "there is a Position still open for long")
                return  
            buysell = "S"
        else:
            logging.error( "takePosition must be either long or short")
            raise Exception( position.takePosition )
        position.buysell = buysell
        
        fmt = "%d.%m.%Y %H:%M:%S"
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowTime = datetime.datetime.now(moscowTimeZone)
        nSec = position.entryTimeSeconds
        moscowTime_plusNsec = moscowTime + datetime.timedelta(seconds = nSec)
        expdate = moscowTime_plusNsec.strftime(fmt) 
        brokerref = union
        quantity = position.quantity
        bymarket = position.bymarket 
        price = round(position.entryPrice , position.decimals)
        price = "{0:0.{prec}f}".format(price,prec=position.decimals)
        
        usecredit = False
        
        res = self.tc.new_order(
            board,seccode,client,union,buysell,expdate,brokerref,
            quantity,price,bymarket,usecredit
        )
        log.debug(repr(res))
        if res.success == True:
            position.entry_id = res.id
            self.monitoredPositions.append(position)
        else:
            logging.error( "position has not been processed by transaq")
        
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
        
                    
            