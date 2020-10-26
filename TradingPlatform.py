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
        self.id = None
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

    def __str__(self):
        msg = ' takePosition='+ self.takePosition 
        msg += ' board=' + self.board 
        msg += ' seccode=' + self.seccode
        msg += ' marketId=' + str(self.marketId)
        msg += ' entryTimeSeconds=' + str(self.entryTimeSeconds)
        msg += ' quantity=' + str(self.quantity)
        msg += ' entryPrice=' + str(self.entryPrice)
        msg += ' exitPrice=' + str(self.exitPrice)
        msg += ' decimals=' + str(self.decimals)
        msg += ' stoploss=' + str(self.stoploss)

        
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
        
        s = order.status
        monitoredPosition = None

        if s == 'matched':
            
            for m in self.monitoredPositions:
                if m.id == order.id: monitoredPosition = m; break;
                     
            if monitoredPosition is None :
                logging.debug( 'already processed: %s', str(monitoredPosition))
                return
                    
            if monitoredPosition.stopOrderRequested == False :
                
                monitoredPosition.stopOrderRequested = True 
                board = order.board
                seccode = order.seccode
                client = order.client
                buysell = ""
                if  monitoredPosition.buysell == "B":
                    buysell = "S"
                elif monitoredPosition.buysell == "S":
                    buysell = "B"
                else:
                    raise Exception("what? " + monitoredPosition.buysell)
                quantity = monitoredPosition.quantity
                trigger_price = monitoredPosition.exitPrice
                correction = 0
                use_credit = False
                
                self.tc.new_takeprofit(
                    board, 
                    seccode, 
                    client, 
                    buysell, 
                    quantity, 
                    trigger_price,
                    correction,
                    use_credit
                )   

            self.monitoredPositions = [ p for p in self.monitoredPositions if p.id != order.id ]            
            
        elif s in ['watching','active','forwarding']:
            
            if not any( mo.id == order.id for mo in self.monitoredOrders):
                self.monitoredOrders.append(order)
                msg = 'order s% with status: %s added to monitoredOrders'
                logging.info( msg, order.id, s )                
           
        elif s in ['rejected','expired','denied'] :
            
            self.monitoredOrders = [ o for o in self.monitoredOrders if o.id != order.id ]
            msg = 'order s% with status: %s deleted from monitoredOrders'
            logging.info( msg , order.id, s )
                    
        else :            
            others = """
                "none","inactive","wait","cancelled","disabled","failed",
                "refused","removed"
            """
            logging.debug( 'status: %s skipped, belongs to: %s', s, others )  

    def processStopOrderStatus(self, stopOrder):
        
        s = stopOrder.status
        
        if s in ['tp_guardtime','tp_forwarding'] :
            if not any(o.id == stopOrder.id for o in self.monitoredStopOrders):
                self.monitoredStopOrders.append(stopOrder)
                msg='stopOrder s% with status: %s added to monitoredStopOrders'
                logging.info( msg, stopOrder.id, s )                


        elif s == 'tp_executed':
            self.monitoredStopOrders = [ 
                o for o in self.monitoredStopOrders if o.id != stopOrder.id
            ]
            msg='takeProfit s% with status: %s deleted from monitoredStopOrders'
            logging.info( msg, stopOrder.id, s )
           
        else:            
            others = """
                "linkwait","sl_forwarding","sl_guardtime","tp_correction",
                "tp_correction_guardtime","watching","tp_forwarding","cancelled"
                "denied","disabled","expired","failed","rejected","sl_executed"
            """
            logging.debug( 'status: %s skipped, belongs to: %s', s, others )  

    def processPosition ( self, position):
        
        board = position.board
        seccode = position.seccode
        client = self.getClientIdByMarket(position.marketId)
        union =  self.getUnionIdByMarket(position.marketId)        
        buysell = ""
        if (position.takePosition == "long"):
            buysell = "B"
        elif (position.takePosition  == "short"):
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
        bymarket = False
        price = round(position.entryPrice , position.decimals)
        price = "{0:0.{prec}f}".format(price,prec=position.decimals)
        #TODO price = 
        usecredit = False
        
        res = self.tc.new_order(
            board,seccode,client,union,buysell,expdate,brokerref,
            quantity,price,bymarket,usecredit
        )
        log.debug(repr(res))
        if res.success == True:
            position.id = res.id
            self.monitoredPositions.append(position)
        else:
            logging.error( "position has not been processed by transaq")
                    
            