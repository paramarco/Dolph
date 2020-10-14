# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:23:46 2019
last edit on Wed Aug 22 10:23:46 2019
@author: mvereda
"""
import logging
import time 
import datetime
import pytz
import threading
import sys
import socket
import sqlite3
import xml.etree.cElementTree as et
import pandas as pd
import csv
from xml.etree.ElementTree import fromstring, ElementTree
import json


log = logging.getLogger("DataServer")


class DataServer:
    def __init__(self):
        self.__host = '127.0.0.1'
        self.__port = 13000
        self.__dataFrames = {}
        
        log.info('DataServer starting ...')

    def __createDBtables(self):
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            cursor = sqliteConnection.cursor()
            log.info(" Connected to database ")
            query = """  BEGIN TRANSACTION; """
            cursor.execute(query)
            
            query = """   DROP TABLE IF EXISTS security; """
            cursor.execute(query)
            
            query = """  DROP TABLE IF EXISTS old_quote """
            cursor.execute(query)
            query = """  DROP TABLE IF EXISTS quote """
            cursor.execute(query)

            query = """
                CREATE TABLE security ( 
                    id INTEGER NOT NULL,
                    CODE TEXT NOT NULL,
                    PERIOD INTEGER NOT NULL,
                    BOARD TEXT NOT NULL                     
                );
            """
            cursor.execute(query)            

            query = """ ALTER TABLE quote RENAME TO old_quote """
            cursor.execute(query)
            query = """
                CREATE TABLE quote (
                    DATE_TIME TEXT NOT NULL,
                    OPEN REAL NOT NULL,
                    HIGH REAL NOT NULL,
                    LOW REAL NOT NULL,
                    CLOSE REAL NOT NULL,
                    VOL INTEGER NOT NULL,
                    security_id INTEGER NOT NULL,
                    CONSTRAINT constraint_time 
                        UNIQUE (DATE_TIME, security_id)
                );
            """
            cursor.execute(query)
            query = """  
                INSERT INTO quote 
                    SELECT DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL,security_id 
                    FROM old_quote 
            """
            query = """  INSERT INTO quote SELECT* FROM old_quote """
            cursor.execute(query)
            query = """  COMMIT;  """
            cursor.execute(query)
            query = """  DROP TABLE old_quote """
            cursor.execute(query)
            query = """  VACUUM; """
            cursor.execute(query)
            
            sqliteConnection.commit()
            cursor.close()
            
        except sqlite3.Error as error:
            log.error("Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                log.info("database closed")
                
    def __normalizeVolume(self):
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            cursor = sqliteConnection.cursor()
            log.info(" Connected to database ") 
            
            update_query = """
                UPDATE quote 
                 SET 
                     VOL = VOL / 10
                WHERE 
                     security_id = X AND
                     DATE_TIME >= '20140101 100000'
            """
            cursor.execute(update_query)
            cursor.execute("commit")
            sqliteConnection.commit()
                
            cursor.close()
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()


    def __deleteOlder(self, date):
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            cursor = sqliteConnection.cursor()
            log.info(" Connected to database ") 
            
            sqlite_delete_query = """
                DELETE FROM
                    quote
                WHERE 
                    DATE >= """ + date + " "
                    
            cursor.execute(sqlite_delete_query)
            cursor.execute("commit")
            sqliteConnection.commit()
                
            cursor.close()
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
        
    def __deleteTableDuplicates(self):
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            cursor = sqliteConnection.cursor()
            log.info(" connected to database ") 
                       
            sqlite_delete_query = """
                DELETE FROM
                     quote
                 WHERE 
                     rowid NOT IN (
                         SELECT min(rowid) 
                         FROM quote 
                         GROUP BY 
                             DATE,
                             TIME
                    )
            """
            cursor.execute(sqlite_delete_query)
            cursor.execute("commit")
            sqliteConnection.commit()
            cursor.close()
    
        except sqlite3.Error as error:
            log.error(" Failed to read data from database table", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                log.info(" finished")
                
    def __showTablesInfo(self, securities ):

        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            cursor = sqliteConnection.cursor()
            log.info(" connected to database")                         
                
            cursor.execute('PRAGMA TABLE_INFO(quote)')
            result=cursor.fetchall()
            log.info(' table quote')
            for row in result:
                log.info(row)
                
            query= """
                SELECT sql 
                FROM sqlite_master 
                WHERE 
                    type='table' AND 
                    name='quote'
            """
            cursor.execute(query)
            schema = cursor.fetchone()
            entries = [ 
                tmp.strip() for tmp in schema[0].splitlines()
                    if tmp.find("constraint")>=0 or tmp.find("unique")>=0
            ]
            for i in entries: log.info(i)
                
            sql_count_query = "select count() from quote"
            cursor.execute(sql_count_query)
            result=cursor.fetchone()
            log.info('number of rows: ' + str(result[0]))        
                
            cursor.close()
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                log.info(" database cleaned-up!")
  
    def __buildQueryFromOneSecCode(self,securities):
        try:             
        
            seccode = securities[0]['seccode']
            
            sqliteConnection = sqlite3.connect('bimbi.sqlite') 
            cursor = sqliteConnection.cursor()  
            query = """
                SELECT id 
                FROM security 
                WHERE 
                    CODE = '"""+securities[0]['seccode']+"""' AND
                    BOARD = '"""+securities[0]['board']+"' "
                    
            cursor.execute(query)
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError("getDataFrames: seccode not found")                        
            security_id = str(result[0])        
            
            fieldsSELECT = """  {0}.DATE_TIME,
                                {0}.OPEN AS {0}_OPEN,
                                {0}.HIGH AS {0}_HIGH,
                                {0}.LOW AS {0}_LOW,
                                {0}.CLOSE AS {0}_CLOSE ,
                                {0}.VOL AS {0}_VOL   """.format(seccode)
            condWHERE_SecIds = str(seccode)+".security_id = "+str(security_id)
            
            query = """
                SELECT """+fieldsSELECT+"""
                FROM quote """+seccode+"""
                WHERE                    
                   """+condWHERE_SecIds+"""
                ORDER BY
                   """+seccode+""".DATE_TIME ASC               
            """
            return query
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
    
    
    def __buildQueryFromSecCodes(self,securities, since):
        
        date = str(since.year) + str(since.month).zfill(2) 
        date +=str(since.day).zfill(2) + ' 000000'
        log.info(" since: " +  date)
        
        if (len(securities) == 1):
            return self.__buildQueryFromOneSecCode(securities)
            
        try:            
            log.info(" connecting to database ...")            
        
            sqliteConnection = sqlite3.connect('bimbi.sqlite') 
            cursor = sqliteConnection.cursor()
            
            seccodes = [s['seccode'] for s in securities]
            securityIds = []
            listSelectFields = []
            for security in securities:
                query = """
                    SELECT id 
                    FROM security 
                    WHERE 
                        CODE = '"""+security['seccode']+"""' AND
                        BOARD = '"""+security['board']+"' "
                        
                cursor.execute(query)                
                result = cursor.fetchone()
                
                errMsg="getDataFrames: "+security['seccode']+" not found"
                if result is None:                    
                    raise RuntimeError(errMsg)
                    
                security_id = str(result[0])
                fields="""{0}.OPEN AS {0}_OPEN,
                          {0}.HIGH AS {0}_HIGH,
                          {0}.LOW AS {0}_LOW,
                          {0}.CLOSE AS {0}_CLOSE ,
                          {0}.VOL AS {0}_VOL   """.format(security['seccode'])
                listSelectFields.append(fields) 
                securityIds.append((security['seccode'],security_id))
                                    
            fieldsSELECT = "   "+seccodes[0]
            trailingSpace = ",\n                            "
            fieldsSELECT += ".DATE_TIME"+trailingSpace
            fieldsSELECT += trailingSpace.join(listSelectFields)            
            
            fieldsFROM = ", ".join(["quote " + s for s in seccodes])
            
            condWHERE_date = ""
            i = 0            
            while i < len(seccodes) - 1 :
              condWHERE_date += seccodes[i]+".DATE_TIME = "
              condWHERE_date += seccodes[i+1]+".DATE_TIME AND\n                   "
              i = i + 1
            condWHERE_date += seccodes[0]+".DATE_TIME > '" + date +"' AND "

              
            condWHERE_SecIds = ""
            for (seccode,idSec) in securityIds[:-1]:
                condWHERE_SecIds += str(seccode)+".security_id = "+str(idSec)
                condWHERE_SecIds += " AND\n                   "
            condWHERE_SecIds += securityIds[-1][0]
            condWHERE_SecIds += ".security_id = "+securityIds[-1][1]
            query = """
                SELECT """+fieldsSELECT+"""
                FROM """+fieldsFROM+"""
                WHERE                    
                   """+condWHERE_date+condWHERE_SecIds+"""
                ORDER BY
                   """+seccodes[0]+""".DATE_TIME ASC               
            """            
            return query
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()


    def __buildQueryGetClosePrice(self, securities, since):  
        
        date = str(since.year) + str(since.month).zfill(2) 
        date +=str(since.day).zfill(2) + ' 000000'
        log.info(" since: " +  date)
            
        try:            
            log.info(" connecting to database ...")            
        
            sqliteConnection = sqlite3.connect('bimbi.sqlite') 
            cursor = sqliteConnection.cursor()
            
            seccodes = [s['seccode'] for s in securities]
            securityIds = []
            listSelectFields = []
            for security in securities:
                query = """
                    SELECT id 
                    FROM security 
                    WHERE 
                        CODE = '"""+security['seccode']+"""' AND
                        BOARD = '"""+security['board']+"' "
                        
                cursor.execute(query)                
                result = cursor.fetchone()
                
                errMsg="GetClosePrice: "+security['seccode']+" not found"
                if result is None:                    
                    raise RuntimeError(errMsg)
                    
                security_id = str(result[0])                
                securityIds.append((security['seccode'],security_id))
            
                
            
            fields="""{0}.CLOSE AS {0}_CLOSE,
                          {0}.OPEN AS {0}_OPEN,
                          {0}.HIGH AS {0}_HIGH,
                          {0}.LOW AS {0}_LOW,
                          {0}.VOL AS {0}_VOL   """.format(securities[0]['seccode'])   
            listSelectFields.append(fields) 
            
            for security in securities[1:]:                        
                fields="{0}.CLOSE AS {0}_CLOSE".format(security['seccode'])
                listSelectFields.append(fields) 
            
            fieldsSELECT = "   "+seccodes[0]
            trailingSpace = ",\n                            "
            fieldsSELECT += ".DATE_TIME AS DATE_TIME"+trailingSpace
            fieldsSELECT += trailingSpace.join(listSelectFields)            
            
            fieldsFROM = ", ".join(["quote " + s for s in seccodes])
            
            condWHERE_date = seccodes[0]+".DATE_TIME > '" + date
            condWHERE_date += "' AND \n                          "
            i = 0            
            while i < len(seccodes) - 1 :
              condWHERE_date += seccodes[i]+".DATE_TIME = "
              condWHERE_date += seccodes[i+1]+".DATE_TIME AND\n                   "
              i = i + 1
                
            condWHERE_SecIds = ""
            for (seccode,idSec) in securityIds[:-1]:
                condWHERE_SecIds += str(seccode)+".security_id = "+str(idSec)
                condWHERE_SecIds += " AND\n                   "
            condWHERE_SecIds += securityIds[-1][0]
            condWHERE_SecIds += ".security_id = "+securityIds[-1][1]
            query = """
                SELECT """+fieldsSELECT+"""
                FROM """+fieldsFROM+"""
                WHERE                    
                   """+condWHERE_date+condWHERE_SecIds+"""
                ORDER BY
                   """+seccodes[0]+""".DATE_TIME ASC               
            """            
            return query
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()

    
    def retrieveData(self, securities,periods,since=None,target = None):
        query = self.__buildQueryGetClosePrice(securities, since)
        log.info(" reading from database ...")
        fmt = '%Y%m%d %H:%M:%S'
        seccodes = [s['seccode'] for s in securities]
        agg_dict = {}
        name_dict = {}
        agg_dict[seccodes[0]+"_CLOSE"] = ['last']
        agg_dict[seccodes[0]+"_OPEN"] = ['first']
        agg_dict[seccodes[0]+"_HIGH"] = ['max']
        agg_dict[seccodes[0]+"_LOW"] = ['min'] 
        agg_dict[seccodes[0]+"_VOL"] = ['sum']
        
        name_dict[seccodes[0]+"_CLOSE"] = None
        name_dict[seccodes[0]+"_OPEN"] = None
        name_dict[seccodes[0]+"_HIGH"] = None
        name_dict[seccodes[0]+"_LOW"] = None
        name_dict[seccodes[0]+"_VOL"] = None

        for seccode in seccodes[1:]:
            agg_dict[seccode+"_CLOSE"] = ['last']
            name_dict[seccode+"_CLOSE"] = None

        try: 
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            data = pd.read_sql_query(
                query,
                sqliteConnection,                
                parse_dates={'DATE_TIME':fmt},
                index_col='DATE_TIME'
            )            
            
            for p in periods:
                log.info(" re-sampling to "+p+" ...")
    
                df = data.resample(p).agg(agg_dict,names=name_dict).dropna()
                df.columns = df.columns.droplevel(1)
                self.__dataFrames[p] = df  
            
            return self.__dataFrames
            
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close(); log.info(" finished")


    def __querySearchSec(self, security, since):        
        
        date = str(since.year) + str(since.month).zfill(2) 
        date +=str(since.day).zfill(2) + ' 000000'
        log.debug(" since: " +  date)
        
        try:            
            sqliteConnection = sqlite3.connect('bimbi.sqlite') 
            cursor = sqliteConnection.cursor()   
            
            query = """
                SELECT id 
                FROM security 
                WHERE 
                    CODE = '"""+security['seccode']+"""' AND
                    BOARD = '"""+security['board']+"' "                    
     
            cursor.execute(query)                
            result = cursor.fetchone()
                
            errMsg = security['seccode']+" not found"
            if result is None:                    
                raise RuntimeError(errMsg)
                    
            security_id = str(result[0]) 
            
            fieldsSELECT =   """'{0}' AS Mnemonic ,
                                  LOW AS MinPrice,
                                  HIGH AS MaxPrice,
                                  OPEN AS StartPrice,
                                  CLOSE AS EndPrice,
                                  1 AS HasTrade,
                                  VOL AS addedVolume,
                                  VOL AS numberOfTrades,
                                  DATE_TIME AS DATE_TIME                        
                          """.format(security['seccode']) 
            
            fieldsFROM = "quote " 
                
            condWHERE = "security_id = " + security_id
            condWHERE += " AND DATE_TIME > '" + date +"' "
            query = """
                SELECT   """+fieldsSELECT+"""
                FROM     """+fieldsFROM+"""
                WHERE    """+condWHERE+"""
                ORDER BY  DATE_TIME ASC               
            """            
            return query
    
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()

    def searchData(
            self, 
            securities, 
            periods, 
            since, 
            target = None, 
            between_time = ('10:00', '18:00')
        ):
        
        log.debug(" searching into database ...")
        fmt = '%Y%m%d %H:%M:%S'
        tb = between_time[0]
        te = between_time[1]
        agg_dict = {
            'Mnemonic' :['last'],
            'MinPrice' : ['min'] ,
            'MaxPrice' : ['max'] ,
            'StartPrice': ['first'] ,
            'EndPrice': ['last'] ,
            'HasTrade': ['last'] ,
            'addedVolume': ['sum'],
            'numberOfTrades': ['sum']                    
        }
        if target is not None:
            securities = [target]
        try:
            dfs = {}
            for p in periods:
                list_df = []
                for sec in securities:
                    log.debug( "building dataframe for: " + str(sec) )
                    query = self.__querySearchSec( sec, since )
                    
                    sqliteConnection = sqlite3.connect('bimbi.sqlite')
                    df = pd.read_sql_query(
                        query,
                        sqliteConnection,                
                        parse_dates={'DATE_TIME':fmt},
                        index_col='DATE_TIME'
                    )  
                    sqliteConnection.close() 
                    df = df.loc[ since :]
                    df = df.between_time(tb, te)
                    df = df.resample(p).agg(agg_dict).dropna()
                    df.columns = df.columns.droplevel(1)
                    list_df.append(df)
                
                df = pd.concat( list_df, axis=0 )
                dfs[p] = df
                
            return dfs
            
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close(); log.debug(" finished")    
    
    def getDataFrames(self):
        return self.__dataFrames
    
    def resampleDataFrames(self, securities, periods = ['1Min'],
                           since=None, target = None):        
        try:            
            query = self.__buildQueryFromSecCodes(securities, since)
            seccodes = [s['seccode'] for s in securities]
            agg_dict = {}
            name_dict = {}
            for seccode in seccodes:
                agg_dict[seccode+"_OPEN"] = ['first']
                agg_dict[seccode+"_HIGH"] = ['max']
                agg_dict[seccode+"_LOW"] = ['min']
                agg_dict[seccode+"_CLOSE"] = ['last']
                agg_dict[seccode+"_VOL"] = ['sum']
                
                name_dict[seccode+"_OPEN"] = None
                name_dict[seccode+"_HIGH"] = None
                name_dict[seccode+"_LOW"] = None
                name_dict[seccode+"_CLOSE"] = None
                name_dict[seccode+"_VOL"] = None

            log.info(" reading from database ...")
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            data = pd.read_sql_query(
                 query,
                 sqliteConnection,
                 parse_dates = 
                     {'DATE_TIME' : '%Y%m%d %H%M%S'},
                 index_col='DATE_TIME'
            )
            for period in periods:
                log.info(" re-sampling to "+period+" ...")
    
                self.__dataFrames[period] = data.resample(period)       \
                                                .agg(agg_dict, 
                                                     names=name_dict)   \
                                                .dropna()
            return self.__dataFrames
        
        except sqlite3.Error as error:
            log.error(" Failed to read from database", error)
        finally:
            if (sqliteConnection):
                sqliteConnection.close()
                log.info(" finished")

                
    def __buffered_readLine(self,s):
        buffer = []
        while True:
            chunk = s.recv(1)
            if chunk == b'':
                raise RuntimeError("buffered_readLine: socket broken")        
            if  chunk != b'\n':
                buffer.append(chunk)
            elif chunk == b'\n':
                break
    
        return " " + b''.join(buffer).decode("utf-8-sig")
    
    def __sendHistoryRequest(self,board,seccode,period,count,reset):
        
        log.info(" requesting archive to TRANSAQ ...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.__host,self.__port))    
            cmd='<command id="gethistorydata">'
            cmd+='<security><board>'+board+'</board>'
            cmd+='<seccode>'+seccode+'</seccode></security>'
            cmd+='<period>'+period+'</period>'
            cmd+='<count>'+count+'</count>'
            cmd+='<reset>'+reset+'</reset>'
            cmd+='</command>\n'
            log.info( cmd )
            s.send(cmd.encode()) 
            s.close()
        except:
            log.error(sys.exc_info()[0])
            raise
            
    def __parse_XML2pandas(self,xml_string, df_cols): 
        
        xtree = ElementTree(fromstring(xml_string))
        xroot = xtree.getroot()
        rows = []
        for node in xroot: 
            res = [] 
            for col in df_cols:
                if node is not None and node.attrib.get(col) is not None:
                    res.append(float(node.attrib.get(col)))
                else: 
                    res.append(None)
            rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})
        
        df_columns = ["open", "close", "high", "low", "volume"]
        df = pd.DataFrame(columns = df_columns)
        df.append(rows, ignore_index=True) 

        return df
    
    def __getSecurityIdSQL (self,board,seccode):
        
        errMsg = "__getSecurityIdSQL: seccode: "+seccode+" not found"
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            sqliteConnection.isolation_level = None
            cursor = sqliteConnection.cursor()
            query = """
                SELECT id 
                FROM security 
                WHERE 
                    CODE = '"""+seccode+"""' AND 
                    BOARD = '"""+board+"""'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result is None:                
                raise RuntimeError(errMsg)
                
            sqliteConnection.close()
            return str(result[0])
    
        except sqlite3.Error as error:
            log.error(errMsg, error)                
    
    def __getCandleValues (self,candle, security_id):
        try:
            if candle.attrib.get('date') is not None:
                xmlDate = candle.attrib.get('date')
                parts = xmlDate.split(' ')
                dateParts = parts[0].split('.')
                dateParts.reverse()
                date = ''.join(dateParts)
                time = ''.join(parts[1].split(':'))
                date_time = date + " " + time
            if candle.attrib.get('open') is not None:
                open = float(candle.attrib.get('open'))
            if candle.attrib.get('high') is not None:
                high = float(candle.attrib.get('high'))
            if candle.attrib.get('low') is not None:
                low = float(candle.attrib.get('low'))
            if candle.attrib.get('close') is not None:
                close = float(candle.attrib.get('close'))
            if candle.attrib.get('volume') is not None:
                vol = float(candle.attrib.get('volume'))
                
            return (date_time,open,high,low,close,vol,security_id)    
        
        except TypeError as error:
            log.error(" Failed with TypeError:", error)         
    
    def __addRow2dataFrame(self, seccode, values):
        if bool(self.__dataFrames):
            (date_time,open,high,low,close,vol,security_id) = values
            t = pd.to_datetime(date_time,format='%Y%m%d %H%M%S')
            
            self.__dataFrames['1Min'].at[t,seccode+"_OPEN"] = open
            self.__dataFrames['1Min'].at[t,seccode+"_HIGH"] = high
            self.__dataFrames['1Min'].at[t,seccode+"_LOW"] = low
            self.__dataFrames['1Min'].at[t,seccode+"_CLOSE"] = close
            self.__dataFrames['1Min'].at[t,seccode+"_VOL"] = vol  
            log.info(" rows updated: " + str(values))    
            
    def __shouldGo2Archive(self, mode, values):
        
        infoMsg = " online data for " + str(values) + " goes to the archive"
        shouldGo = True        
        if (mode == "online"):            
            moscowTimeZone = pytz.timezone('Europe/Moscow')                    
            moscowDateTime = datetime.datetime.now(moscowTimeZone)
            stringMoscowMinute =  str(moscowDateTime.year).zfill(4) 
            stringMoscowMinute += str(moscowDateTime.month).zfill(2) 
            stringMoscowMinute += str(moscowDateTime.day).zfill(2) + " " 
            stringMoscowMinute += str(moscowDateTime.hour).zfill(2) 
            stringMoscowMinute += str(moscowDateTime.minute).zfill(2) + "00"
            
            shouldGo = True if ( stringMoscowMinute >  values[0]) else False
            
            if shouldGo == True: log.info( infoMsg )
                
        return shouldGo
    
    def __parse_XML2sql(self,xml_string, mode):
        try:        
            xtree = ElementTree(fromstring(xml_string))
            xroot = xtree.getroot()
            if xroot.attrib.get('board') is not None:
                board = xroot.attrib.get('board')
            if xroot.attrib.get('seccode') is not None:
                seccode = xroot.attrib.get('seccode')

            security_id = self.__getSecurityIdSQL(board,seccode)
            
            log.info(" writing to database ...")            
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            sqliteConnection.isolation_level = None
            cursor = sqliteConnection.cursor()
            cursor.execute("begin") 
            
            for candle in xroot.findall('candle'):                        
                
                values = self.__getCandleValues(candle,security_id)
                
                if not (self.__shouldGo2Archive (mode, values )):
                    continue
                   
                query_insert =  """
                 INSERT OR IGNORE INTO quote 
                 ('DATE_TIME','OPEN','HIGH','LOW','CLOSE','VOL','security_id') 
                 VALUES (?,?,?,?,?,?,?) 
                """
                query_update =  """
                    UPDATE quote SET 
                        OPEN = '"""          + str(values[1]) + """',
                        HIGH = '"""          + str(values[2]) + """',
                        LOW = '"""           + str(values[3]) + """',
                        CLOSE = '"""         + str(values[4]) + """',
                        VOL = '"""           + str(values[5]) + """'
                    WHERE
                        DATE_TIME = '"""     + str(values[0])+ """' AND
                        security_id = '"""   + str(values[6]) + """'
                """ 
                try:
                    cursor.execute(query_insert, values)
                    cursor.execute(query_update)
                except sqlite3.Error as error:
                    log.error(" Failed for insert", error )
                    log.error(" VALUES:  ", values )
                
            cursor.execute("commit")
            sqliteConnection.commit()
            sqliteConnection.close()
              
        except:
            log.error(" Failed to read from XML")
    
    def isConsolidatedCandle (self, candle):
        infoMsg = "yes, it  goes to the archive"
        infoMsgNot = "no, isn't consolidated yet, not a minute old yet"
        fmt = "%Y%m%d %H%M%S"

        shouldGo = False           
        
        moscowTimeZone = pytz.timezone('Europe/Moscow')                    
        moscowDateTime = datetime.datetime.now(moscowTimeZone)
        moscowDateTime_minus1min = moscowDateTime - datetime.timedelta(minutes = 1)
        
        t1 = 'moscowtime now:  '+ moscowDateTime.strftime(fmt) 
        t2 = 'moscowtime -1min:'+ moscowDateTime_minus1min.strftime(fmt)
        t3 = 'Candletime:      '+ candle.date.strftime(fmt)         
        log.debug(t1) ; log.debug(t2); log.debug(t3);
        
        if ( moscowDateTime_minus1min >  moscowTimeZone.localize( candle.date ) ):
            shouldGo = True 
            log.debug( infoMsg )
        else:
            log.debug( infoMsgNot )

        return shouldGo

    def datetime2SQLString(self, datetime):
        stringDatetime =  str(datetime.year).zfill(4) 
        stringDatetime += str(datetime.month).zfill(2) 
        stringDatetime += str(datetime.day).zfill(2) + " " 
        stringDatetime += str(datetime.hour).zfill(2) 
        stringDatetime += str(datetime.minute).zfill(2) + "00"
        return stringDatetime
        
    def storeCandles (self, historyCandlePacket ):
        
        board = historyCandlePacket.board
        seccode = historyCandlePacket.seccode
        security_id = self.__getSecurityIdSQL(board, seccode)

        try:            
            sqliteConnection = sqlite3.connect('bimbi.sqlite')
            sqliteConnection.isolation_level = None
            cursor = sqliteConnection.cursor()
            cursor.execute("begin") 
            
            for c in historyCandlePacket.items:
                
                if not (self.isConsolidatedCandle( c ) ):
                    continue
                
                values = ( 
                    self.datetime2SQLString(c.date), 
                    str(c.open), 
                    str(c.high),
                    str(c.low), 
                    str(c.close), 
                    str(c.volume), 
                    security_id
                )                
                query_insert =  """
                 INSERT OR IGNORE INTO quote 
                 ('DATE_TIME','OPEN','HIGH','LOW','CLOSE','VOL','security_id') 
                 VALUES (?,?,?,?,?,?,?) 
                """
                query_update =  """
                    UPDATE quote SET 
                        OPEN = '"""          + values[1] + """',
                        HIGH = '"""          + values[2] + """',
                        LOW = '"""           + values[3] + """',
                        CLOSE = '"""         + values[4] + """',
                        VOL = '"""           + values[5] + """'
                    WHERE
                        DATE_TIME = '"""     + values[0] + """' AND
                        security_id = '"""   + values[6] + """'
                """ 
                try:
                    cursor.execute(query_insert, values)
                    cursor.execute(query_update)
                except sqlite3.Error as error:
                    log.error(" Failed to insert", error )
                    log.error(" VALUES:  ", values )
                
            cursor.execute("commit")
            sqliteConnection.commit()
            sqliteConnection.close()
        except Exception as inst:
            errMsg = str(type(inst))+ "\n"+ str(inst.args) +"\n"+ str(inst)
            log.error(errMsg)
            log.error(" Failed to commit")
            
    
    def __parse_XMLfile2pandas(self,xml_file, df_cols):
        xtree = et.parse(xml_file)
        xroot = xtree.getroot()
        rows = []
        for node in xroot: 
            res = [] 
            for col in df_cols:
                if node is not None and node.attrib.get(col) is not None:
                    res.append(node.attrib.get(col))
                else: 
                    res.append(None)
            rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})
        
        out_df = pd.DataFrame(rows, columns=df_cols)        
        return out_df
    
    def __updateDBquotesLoop(self,securities):
        while True:
            time.sleep(5)
            self.updateDBquotes(securities, 'online')
            self.resampleDataFrames( securities )
    
    def syncDB(self, securities, online=False ):
        self.updateDBquotes(securities, 'archive')
        self.resampleDataFrames( securities )
        if online:
            syncThread = threading.Thread(target = self.__updateDBquotesLoop,
                                          args = (securities,))
            syncThread.start()            
    
    def __readHistoryCandles (self, s, mode):
        matches = ['status="2"', 'status="1"', 'status="3"']
        lastCandle = False
        data = ''
        try: 
            while True:
                line = self.__buffered_readLine(s).strip()       
                data += line
                if 'candles' in line:
                    if any(x in line for x in matches):
                        if mode == 'online':
                            lastCandle = True
                    elif 'status="0"' in line:
                        lastCandle = True
                    else:
                        break
            return lastCandle, data
        
        except Exception as inst:
            errMsg = 'ERROR :: __readHistoryCandles: ' + str(type(inst)) 
            errMsg += "\n"+ str(inst.args) +"\n"+ str(inst)
            log.error(errMsg)
    
    def updateDBquotes(self, securities, mode='archive'): 
        log.info(" updating DataBase ...")
        count = '33000'
        reset = 'true'
        if mode == 'online':
            count = '2'
        try:            
            for security in securities:                
                self.__sendHistoryRequest(security['board'],
                                          security['seccode'],
                                          '1', count, reset )                
                lastCandle = False               
                while not lastCandle:
                    data = ''
                    time.sleep(2)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((self.__host,self.__port))
                    s.send('<sendHistoryCandle>\n'.encode())                     
                    lastCandle, data = self.__readHistoryCandles(s, mode)                    
                    self.__parse_XML2sql( data, mode )
                    log.info(' got this from transaq: '+data)
                    s.close()
                    
        except Exception as inst:
            errMsg = 'ERROR :: updateDBquotes: ' + str(type(inst)) 
            errMsg += "\n"+ str(inst.args) +"\n"+ str(inst)
            log.error(errMsg)

    def insertCSV2DB(self, security_id, fileName):
        
        log.info(" having security_id: " + str(security_id))
        
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')    
            cursor = sqliteConnection.cursor()
            with open (fileName, 'r') as f:
                reader = csv.reader(f)
                next(reader) 
                         
                query_insert =  """
                 INSERT OR IGNORE INTO quote 
                 ('DATE_TIME','OPEN','HIGH','LOW','CLOSE','VOL','security_id') 
                  VALUES (?,?,?,?,?,?,?)
                """        
                cursor = sqliteConnection.cursor()
                for data in reader:
                    data = data[2:] #TODO does only work for internet DB
                    data[0:2] = [' '.join(data[0 : 2])]
                    data.append(security_id)
                    
                    query_update =  """  
                        UPDATE quote SET 
                            OPEN = '"""          + str(data[1]) + """',
                            HIGH = '"""          + str(data[2]) + """',
                            LOW = '"""           + str(data[3]) + """',
                            CLOSE = '"""         + str(data[4]) + """',
                            VOL = '"""           + str(data[5]) + """'
                        WHERE
                            DATE_TIME = '"""     + str(data[0])+ """' AND
                            security_id = '"""   + str(data[6]) + """'
                    """                     
                    try:
                        cursor.execute(query_insert, data)
                        cursor.execute(query_update)
                    except sqlite3.Error as error:            
                        log.error(" Failed for insert", error )
                        
                sqliteConnection.commit()
            
            log.info(" finished")
                
        except sqlite3.Error as error:            
            log.error(" Failed for insert", error )
        except Exception as inst:
            log.error(str(inst))
            
    def __takeDataFomAnotherSec(self):
        new_security_id = 7
        previous_security_id = 4
        
        log.info(" new_security_id: " + str(new_security_id))
        log.info(" previous_security_id: " + str(previous_security_id))

        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')    
            cursor = sqliteConnection.cursor()
                         
            query =  """
            INSERT OR IGNORE INTO quote 
             ('DATE_TIME','OPEN','HIGH','LOW','CLOSE','VOL','security_id') 
             SELECT
              DATE_TIME,OPEN,HIGH,LOW,CLOSE,VOL,'"""+str(new_security_id)+"""' 
             FROM quote
             WHERE
              security_id = '"""+str(previous_security_id)+"""'
            """        
            cursor = sqliteConnection.cursor()               
                    
            try:
                cursor.execute(query)
                sqliteConnection.commit()
                
            except sqlite3.Error as error:            
                log.error(" Failed for insert", error )
            
            log.info(" finished")
                
        except sqlite3.Error as error:            
            log.error("Failed for insert", error )
        except Exception as inst:
            log.error( str(inst.args))
    
    
    def getSecurityInfo(self, security):        
        
        errMsg = "__getSecurityIdSQL: "+security['seccode']+" not found"
        
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')    
            cursor = sqliteConnection.cursor()
                         
            query = """
                SELECT DECIMALS, MARKET
                FROM security 
                WHERE 
                    CODE = '"""+security['seccode']+"""' AND 
                    BOARD = '"""+security['board']+"""'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result is None:                
                raise RuntimeError(errMsg)
                
            sqliteConnection.close()   
            
            return result[0], result[1]
                
        except sqlite3.Error as error:            
            log.error("Failed for read", error )
        except Exception as inst:
            log.error( str(inst.args))
    
    def getSecurityAlgParams(self, security ):
        
        errMsg = security['seccode']+" not found"
        
        try:
            sqliteConnection = sqlite3.connect('bimbi.sqlite')    
            cursor = sqliteConnection.cursor()
                         
            query = """
                SELECT ALG_PARAMETERS
                FROM security 
                WHERE 
                    CODE = '"""+security['seccode']+"""' AND 
                    BOARD = '"""+security['board']+"""'
            """
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result is None:                
                raise RuntimeError(errMsg)
            sqliteConnection.close()   
                
            params = json.loads(result[0])            
            return params        
        
        except sqlite3.Error as error:            
            log.error("Failed for read", error )
        except Exception as inst:
            log.error( str(inst.args))
    

    # def sendPosition2Transaq(self, position):
      
    #     security = {}
    #     security['seccode'] = position['seccode']
    #     security['board'] = position['board']
    #     decimals, market = self.__getSecurityInfo (security)
    #     buysell = ""
    #     if (position['takePosition'] == "long"):
    #         buysell = "B"
    #     elif (position['takePosition'] == "short"):
    #         buysell = "S"
    #     else:
    #         log.error( "takePosition is either long or short")
    #         return
        
    #     cmd = "<position>";
    #     cmd += "<board>"        + position['board'] + "</board>";
    #     cmd += "<seccode>"      + position['seccode'] + "</seccode>";
    #     cmd += "<client>X</client>";
    #     cmd += "<union>Y</union>";
    #     cmd += "<price>"        + str(position['entryPrice']) + "</price>";
    #     cmd += "<quantity>"     + str(position['quantity']) + "</quantity>";
    #     cmd += "<buysell>"      + buysell + "</buysell>";
    #     cmd += "<stopLoss>"     + str(position['stopLoss']) + "</stopLoss>";
    #     cmd += "<takeProfit>"   + str(position['exitPrice'])+"</takeProfit>";
    #     cmd += "<decimals>"     + str(decimals) + "</decimals>";
    #     cmd += "<market>"       + str(market)+ "</market>";
    #     cmd += "<entryTimeSeconds>" + str(position['entryTimeSeconds']) 
    #     cmd += "</entryTimeSeconds>";        
    #     cmd += "</position>\n";       
        
    #     try:
    #         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         s.connect((self.__host,self.__port))
    #         log.info( cmd )
    #         s.send(cmd.encode()) 
    #         s.close()
    #     except:
    #         log.error(sys.exc_info()[0])            
        
                        
if __name__== "__main__":
    
     connector = DataServer()
     security_id = 11
     fileName = './bkp/ROSN/ROSN/ROSN.csv'
     connector.insertCSV2DB(security_id, fileName)
    
    # fileName = './bkp/SPFB.SBRF/SPFB.SBRF-9.20_190101_200726.csv'
    # connector.insertCSV2DB(security_id, fileName)
    
    
    
    # securities = [] 
   
    
    # securities.append( {'board':'FUT', 'seccode':'SPU0'} )
    # securities.append( {'board':'TQBR', 'seccode':'SBER'} )
    # securities.append( {'board':'TQBR', 'seccode':'GAZP'} )
#    securities.append( {'board':'INDEXM', 'seccode':'RTSI_TQBR'} )
#    connector.updateDBquotes(securities, 'archive')
        
#    period = '2Min'
#
#    while True:            
#        connector.updateDBquotes(securities, 'online')
#        connector.resampleDataFrames(securities, period )
#        dataFrames = connector.getDataFrames()
#        log.info(dataFrames[period].tail())
    
    
    
# takePosition:= long | short | no-go
# entryPrice := price you want to buy if it is long, you want to sell if short
# entryTimeSeconds := cancel position if it is not executed withinin this seconds 
# exitPrice := price you want to re-sell if it is long, you want to buy if short
# stoploss := price you want to exit if your bet was wrong
# quantity := number of lots

    # position = {
    #     'board':'FUT',
    #     'seccode':'SPU0',
    #     'takePosition': 'long',
    #     'entryPrice': 32323.34,
    #     'exitPrice': 3333.3333332,
    #     'stopLoss' : 332323.222222,
    #     'entryTimeSeconds' : 180,
    #     'quantity' : 3
    # }
    # connector.sendPosition2Transaq(position)
    