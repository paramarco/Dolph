# -*- coding: utf-8 -*-
"""
Модуль с основными командами Транзак Коннектора
(см. http://www.finam.ru/howtotrade/tconnector/).

.. note::
    Практически все команды асинхронны!
    Это означает что они возвращают структуру CmdResult, которая говорит лишь
    об успешности отправки команды на сервер, но не её исполнения там.
    Результат исполнения приходит позже в зарегистрированный командой *initialize()* хэндлер.
    Синхронные вспомогательные команды помечены отдельно.
"""
import logging
import ctypes
import platform, os, sys
import lxml.etree as et
from . import structures as ts


log = logging.getLogger("TransaqConnector")

# Check the platform and set the appropriate callback function type
if platform.system() == "Windows":
    callback_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
else:
    callback_func = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)

encoding = sys.stdout.encoding
global_handler = None
timeformat = "%d.%m.%Y %H:%M:%S"

class TransaqException(Exception):
    """
    Класс исключений, связанных с коннектором.
    """
    pass

class TransaqConnector:
    def __init__(self ):
        
        path = ""
        if __file__ is not None:
            path = os.path.dirname(__file__)
            if path != "":
                path += os.sep
        self.txml_dll = ctypes.WinDLL(path + ("txmlconnector64.dll" if platform.machine() == 'AMD64' else 'txmlconnector.dll') )
        

    @callback_func
    def callback( msg):
        """
        Функция, вызываемая коннектором при входящих сообщениях.
    
        :param msg:
            Входящее сообщение Транзака.
        :return:
            True если все обработал.
        """
        obj = ts.parse(msg.decode('utf8'))
        if isinstance(obj, ts.Error):
            log.error("trouble: %s", obj.text)
            raise TransaqException(obj.text.encode(encoding))
        elif isinstance(obj, ts.ServerStatus):
            log.info("connected to server: %s", obj.connected)
            if obj.connected == 'error':
                log.warn("upps, error on connecting: %s", obj.text)
            log.info( repr(obj) )

        if global_handler:
            global_handler(obj)
        return True

    def __get_message(self, ptr):
        try:
            ## Достать сообщение из нативной памяти.
            # msg = ctypes.string_at(ptr)
            # txml_dll.FreeMemory(ptr)
            # return unicode(msg, 'utf8')
            
            return ptr.decode('utf-8')
        
        except Exception as inst:
            errMsg = str(type(inst))+ "\n"+ str(inst.args) +"\n"+ str(inst)
            log.error(errMsg)
            return ptr
        
    
    
    def __elem(self, tag, text):
        # Создать элемент с заданным текстом.
        elem = et.Element(tag)
        elem.text = text
        return elem
    
    
    def __send_command(self, cmd):
        # Отправить команду и проверить на ошибки.
        # msg = self.__get_message(self.txml_dll.SendCommand(cmd))
        sendCmd = self.txml_dll.SendCommand
        sendCmd.restype = ctypes.c_char_p    
        msg = self.__get_message(sendCmd(cmd))
        err = ts.Error.parse(msg)

        if err.text:
            raise TransaqException(err.text.encode(encoding))
        else:
            return ts.CmdResult.parse(msg)
    
    
    def initialize(self, logdir, loglevel, msg_handler):
        """
        Инициализация коннектора (синхронная).
    
        :param logdir:
        :param loglevel:
        :param msg_handler:
        """
        global global_handler
        global_handler = msg_handler
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        # err = self.txml_dll.Initialize(logdir + "\0", loglevel)
        b_logdir = logdir.encode('utf-8')
        myerr = self.txml_dll.Initialize( b_logdir , loglevel)
        if myerr != 0:
            msg = self.__get_message(myerr)
            raise TransaqException(ts.Error.parse(msg).text.encode(encoding))
        if not self.txml_dll.SetCallback(self.callback):
            raise TransaqException(u"Коллбэк не установился")
    
    
    def uninitialize(self ):
        """
        Де-инициализация коннектора (синхронная).
    
        :return:
        """
        # if connected:
        #     self.disconnect()
        self.disconnect()
        err = self.txml_dll.UnInitialize()
        if err != 0:
            msg = self.__get_message(err)
            raise TransaqException(ts.Error.parse(msg).text.encode(encoding))
    
    
    def connect(self, login, password, server, min_delay=100):
        host, port = server.split(':')
        root = et.Element("command", {"id": "connect"})
        root.append(self.__elem("login", login))
        root.append(self.__elem("password", password))
        root.append(self.__elem("host", host))
        root.append(self.__elem("port", port))
        # root.append(self.__elem("rqdelay", str(min_delay)))
       
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def disconnect(self ):
        # global connected
        root = et.Element("command", {"id": "disconnect"})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
        # connected = False
    
    
    def server_status(self):
        root = et.Element("command", {"id": "server_status"})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_instruments(self ):
        root = et.Element("command", {"id": "get_securities"})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def __subscribe_helper(self, board, tickers, cmd, mode):
        root = et.Element("command", {"id": cmd})
        trades = et.Element(mode)
        for t in tickers:
            sec = et.Element("security")
            sec.append(self.__elem("board", board))
            sec.append(self.__elem("seccode", t))
            trades.append(sec)
        root.append(trades)
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def subscribe_ticks(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "subscribe", "alltrades")
    
    
    def unsubscribe_ticks(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "unsubscribe", "alltrades")
    
    
    def subscribe_quotations(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "subscribe", "quotations")
    
    
    def unsubscribe_quotations(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "unsubscribe", "quotations")
    
    
    def subscribe_bidasks(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "subscribe", "quotes")
    
    
    def unsubscribe_bidasks(self, board, tickers):
        return self.__subscribe_helper(board, tickers, "unsubscribe", "quotes")
    
    
    def new_order(self, board, ticker, client, union, buysell, expdate, 
                  brokerref, quantity, price=0, bymarket=True, usecredit=True):
        # Add hidden, unfilled, nosplit
        root = et.Element("command", {"id": "neworder"})
        sec = et.Element("security")
        sec.append(self.__elem("board", board))
        sec.append(self.__elem("seccode", ticker))
        root.append(sec)
        root.append(self.__elem("client", client))
        root.append(self.__elem("union", union))
        root.append(self.__elem("buysell", buysell.upper()))
        root.append(self.__elem("expdate", expdate))
        root.append(self.__elem("brokerref", brokerref))
        # root.append(self.__elem("unfilled", "IOC".encode('utf-8') ) )        
        root.append(self.__elem("quantity", str(quantity)))
        if not bymarket:
            root.append(self.__elem("price", str(price)))
        else:
            root.append(et.Element("bymarket"))
        if usecredit:
            root.append(et.Element("usecredit"))
            
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    def newExitOrder(self, board, ticker, client, buysell, quantity, 
                      trigger_price_sl, trigger_price_tp, correction=0,spread=0, 
                      bymarket=True, usecredit=True, linked_order=None, 
                      valid_for=None):
        
        root = et.Element("command", {"id": "newstoporder"})
        sec = et.Element("security")
        sec.append(self.__elem("board", board))
        sec.append(self.__elem("seccode", ticker))
        root.append(sec)
        root.append(self.__elem("client", client))
        root.append(self.__elem("buysell", buysell.upper()))
        if linked_order:
            root.append(self.__elem("linkedorderno", str(linked_order)))
        if valid_for:
            root.append(self.__elem("validfor", valid_for.strftime(timeformat)))
    
        sl = et.Element("stoploss")
        sl.append(self.__elem("quantity", str(quantity)))
        sl.append(self.__elem("activationprice", str(trigger_price_sl)))
        if not bymarket:
            sl.append(self.__elem("orderprice", str(trigger_price_sl)))
        else:
            sl.append(et.Element("bymarket"))
        if usecredit:
            sl.append(et.Element("usecredit")) 
        sl.append(self.__elem("brokerref","Dolph_sl".encode('utf-8')))   
        root.append(sl)
        
        tp = et.Element("takeprofit")
        tp.append(self.__elem("quantity", str(quantity)))
        tp.append(self.__elem("activationprice", str(trigger_price_tp)))
        # tp.append(et.Element("bymarket"))
        if usecredit:
            tp.append(et.Element("usecredit"))
        tp.append(self.__elem("brokerref","Dolph_tp".encode('utf-8')))
        if correction:
            tp.append(self.__elem("correction", str(correction)))
        if spread:
            tp.append(self.__elem("spread", str(spread)))            
        root.append(tp)
        
        return self.__send_command(et.tostring(root, encoding="utf-8"))

    
    
    def new_stoploss(self, board, ticker, client, buysell, quantity, trigger_price, price=0,
                     bymarket=True, usecredit=True, linked_order=None, valid_for=None):
        root = et.Element("command", {"id": "newstoporder"})
        sec = et.Element("security")
        sec.append(self.__elem("board", board))
        sec.append(self.__elem("seccode", ticker))
        root.append(sec)
        root.append(self.__elem("client", client))
        root.append(self.__elem("buysell", buysell.upper()))
        if linked_order:
            root.append(self.__elem("linkedorderno", str(linked_order)))
        if valid_for:
            root.append(self.__elem("validfor", valid_for.strftime(timeformat)))
    
        sl = et.Element("stoploss")
        sl.append(self.__elem("quantity", str(quantity)))
        sl.append(self.__elem("activationprice", str(trigger_price)))
        if not bymarket:
            sl.append(self.__elem("orderprice", str(price)))
        else:
            sl.append(et.Element("bymarket"))
        if usecredit:
            sl.append(et.Element("usecredit"))
    
        root.append(sl)
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def new_takeprofit(self, board, ticker, client, buysell, quantity, trigger_price,
                       correction=0, use_credit=True, linked_order=None, valid_for=None):
        root = et.Element("command", {"id": "newstoporder"})
        sec = et.Element("security")
        sec.append(self.__elem("board", board))
        sec.append(self.__elem("seccode", ticker))
        root.append(sec)
        root.append(self.__elem("client", client))
        root.append(self.__elem("buysell", buysell.upper()))
        if linked_order:
            root.append(self.__elem("linkedorderno", str(linked_order)))
        if valid_for:
            root.append(self.__elem("validfor", valid_for.strftime(timeformat)))
    
        tp = et.Element("takeprofit")
        tp.append(self.__elem("quantity", str(quantity)))
        tp.append(self.__elem("activationprice", str(trigger_price)))
        # tp.append(et.Element("bymarket"))
        if use_credit:
            tp.append(et.Element("usecredit"))
        if correction:
            tp.append(self.__elem("correction", str(correction)))
        root.append(tp)
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def cancel_order(self, id):
        root = et.Element("command", {"id": "cancelorder"})
        root.append(self.__elem("transactionid", str(id)))
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def cancel_stoploss(self, id):
        root = et.Element("command", {"id": "cancelstoporder"})
        root.append(self.__elem("transactionid", str(id)))
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def cancel_takeprofit(self, id):
        self.cancel_stoploss(id)
    
    
    def get_portfolio(self, client):
        root = et.Element("command", {"id": "get_portfolio", "client": client})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_markets(self):
        """
        Получить список рынков.
    
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "get_markets"})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_history(self, board, seccode, period, count, reset=True):
        """
        Выдать последние N свечей заданного периода, по заданному инструменту.
    
        :param board:
            Идентификатор режима торгов.
        :param seccode:
            Код инструмента.
        :param period:
            Идентификатор периода.
        :param count:
            Количество свечей.
        :param reset:
            Параметр reset="true" говорит, что нужно выдавать самые свежие данные, в
            противном случае будут выданы свечи в продолжение предыдущего запроса.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "gethistorydata"})
        sec = et.Element("security")
        sec.append(self.__elem("board", board))
        sec.append(self.__elem("seccode", seccode))
        root.append(sec)
        root.append(self.__elem("period", str(period)))
        root.append(self.__elem("count", str(count)))
        root.append(self.__elem("reset", "true" if reset else "false"))
       
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    # TODO Доделать условные заявки
    def new_condorder(self, board, ticker, client, buysell, quantity, price,
                      cond_type, cond_val, valid_after, valid_before,
                      bymarket=True, usecredit=True):
        """
        Новая условная заявка.
    
        :param board:
        :param ticker:
        :param client:
        :param buysell:
        :param quantity:
        :param price:
        :param cond_type:
        :param cond_val:
        :param valid_after:
        :param valid_before:
        :param bymarket:
        :param usecredit:
        :return:
        """
        # root = et.Element("command", {"id": "newcondorder"})
        return NotImplemented
    
    
    def get_forts_position(self, client):
        """
        Запрос позиций клиента по FORTS.
    
        :param client:
            Идентификатор клиента.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "get_forts_position", "client": client})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_limits_forts(self, client):
        """
        Запрос лимитов клиента ФОРТС.
    
        :param client:
            Идентификатор клиента.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "get_client_limits", "client": client})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_servtime_diff(self):
        """
        Получить разницу между серверным временем и временем на компьютере пользователя (синхронная).
    
        :return:
            Результат команды с разницей времени.
        """
        return NotImplemented
    
    
    def change_pass(self, oldpass, newpass):
        """
        Смена пароля (синхронная).
    
        :param oldpass:
            Старый пароль.
        :param newpass:
            Новый пароль.
        :return:
            Результат команды.
        """
        root = et.Element("command", {"id": "change_pass", "oldpass": oldpass, "newpass": newpass})
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_version(self):
        """
        Получить версию коннектора (синхронная).
    
        :return:
            Версия коннектора.
        """
        root = et.Element("command", {"id": "get_connector_version"})
        return ts.ConnectorVersion.parse(
            self.__get_message(
                self.txml_dll.SendCommand(et.tostring(root, encoding="utf-8"))
            )
        ).version
    
    
    def get_sec_info(self,market, seccode):
        """
        Запрос на получение информации по инструменту.
    
        :param market:
            Внутренний код рынка.
        :param seccode:
            Код инструмента.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "get_securities_info"})
        sec = et.Element("security")
        sec.append(self.__elem("market", str(market)))
        sec.append(self.__elem("seccode", seccode))
        root.append(sec)
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def move_order(self, id, price, quantity=0, moveflag=0):
        """
        Отредактировать заявку.
    
        :param id:
            Идентификатор заменяемой заявки FORTS.
        :param price:
            Цена.
        :param quantity:
            Количество, лотов.
        :param moveflag:
            0: не менять количество;
            1: изменить количество;
            2: при несовпадении количества с текущим – снять заявку.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "moveorder"})
        root.append(self.__elem("transactionid", str(id)))
        root.append(self.__elem("price", str(price)))
        root.append(self.__elem("quantity", str(quantity)))
        root.append(self.__elem("moveflag", str(moveflag)))
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_limits_tplus(self, client, securities):
        """
        Получить лимиты Т+.
    
        :param client:
            Идентификатор клиента.
        :param securities:
            Список пар (market, seccode) на которые нужны лимиты.
        :return:
            Результат отправки команды.
        """
        root = et.Element("command", {"id": "get_max_buy_sell_tplus", "client": client})
        for (market, code) in securities:
            sec = et.Element("security")
            sec.append(self.__elem("market", str(market)))
            sec.append(self.__elem("seccode", code))
            root.append(sec)
        return self.__send_command(et.tostring(root, encoding="utf-8"))
    
    
    def get_portfolio_mct(self, client):
        """
        Получить портфель МСТ/ММА. Не реализован пока.
    
        :param client:
            Идентификатор клиента.
        :return:
            Результат отправки команды.
        """
        return NotImplemented
    
    def get_united_portfolio(self, client, union=None):
        """
        Получить единый портфель.
        В команде необходимо задать только один из параметров (client или union).
    
        :param client:
            Идентификатор клиента.
        :param union:
            Идентификатор юниона.
        :return:
            Результат отправки команды.
        """
        params = {"id": "get_united_portfolio"}
        if client is not None:
            params["client"] = client
        elif union is not None:
            params["union"] = union
        else:
            raise ValueError("please specify client OR union")
        root = et.Element("command", params)
        return self.__send_command(et.tostring(root, encoding="utf-8"))