import os
import logging
import time
import datetime as dt
import pytz
import threading
import sys
import socket
import psycopg2
from xml.etree.ElementTree import fromstring, ElementTree
import xml.etree.ElementTree as et
import pandas as pd
import csv
import json
from Configuration import Conf as cm

log = logging.getLogger("DataServer")


class DataServer:
    def __init__(self):

        self.__dataFrames = {}
        self._init_configuration()

        log.info('DataServer starting ...')

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
        self.current_tz = cm.current_tz
        self.lastUpdate = None

    def _init_securities(self):
        for sec in self.securities:
            sec['params'] = self.getSecurityAlgParams(sec)

    def __createDBtables(self):
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info("Connected to PostgreSQL database")
            cursor.execute("BEGIN TRANSACTION;")

            cursor.execute("DROP TABLE IF EXISTS security;")
            cursor.execute("DROP TABLE IF EXISTS quote;")

            cursor.execute("""
               CREATE TABLE IF NOT EXISTS public.security
                (
                    id integer NOT NULL DEFAULT nextval('security_id_seq'::regclass),
                    code text COLLATE pg_catalog."default" NOT NULL,
                    period integer NOT NULL,
                    board text COLLATE pg_catalog."default" NOT NULL,
                    decimals integer,
                    market text COLLATE pg_catalog."default",
                    alg_parameters jsonb,
                    platform text COLLATE pg_catalog."default",
                    CONSTRAINT security_pkey PRIMARY KEY (id),
                    CONSTRAINT unique_code UNIQUE (code)
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.quote
                (
                    date_time timestamp with time zone NOT NULL,
                    open real NOT NULL,
                    high real NOT NULL,
                    low real NOT NULL,
                    close real NOT NULL,
                    vol integer NOT NULL,
                    security_id integer,
                    CONSTRAINT constraint_time UNIQUE (date_time, security_id),
                    CONSTRAINT quote_security_id_fkey FOREIGN KEY (security_id)
                        REFERENCES public.security (id) MATCH SIMPLE
                        ON UPDATE NO ACTION
                        ON DELETE NO ACTION
                );
            """)

            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")

    def __normalizeVolume(self, security_id):
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info("Connected to PostgreSQL database")

            cursor.execute("""
                UPDATE quote 
                SET VOL = VOL / 10
                WHERE security_id = %s AND DATE_TIME >= '2014-01-01 10:00:00+00'
            """, (security_id,))

            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")

    def __deleteOlder(self, date):
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info("Connected to PostgreSQL database")

            cursor.execute("""
                DELETE FROM quote
                WHERE DATE_TIME >= %s
            """, (date,))

            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")

    def __deleteTableDuplicates(self):
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info("Connected to PostgreSQL database")

            cursor.execute("""
                DELETE FROM quote
                WHERE ctid NOT IN (
                    SELECT min(ctid) 
                    FROM quote 
                    GROUP BY DATE_TIME, security_id
                )
            """)

            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")


    def __showTablesInfo(self):
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info("Connected to PostgreSQL database")

            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'quote'
            """)
            result = cursor.fetchall()
            log.info('Table quote schema:')
            for row in result:
                log.info(row)

            cursor.execute("SELECT COUNT(*) FROM quote")
            result = cursor.fetchone()
            log.info(f'Number of rows: {result[0]}')

            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")

    def __buildQueryFromOneSecCode(self, securities):
        try:
            seccode = securities[0]['seccode']
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            query = """
                SELECT id 
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (securities[0]['seccode'], securities[0]['board']))
            result = cursor.fetchone()
            if result is None:
                raise RuntimeError("getDataFrames: seccode not found")
            security_id = str(result[0])

            fieldsSELECT = f"""
                {seccode}.DATE_TIME,
                {seccode}.OPEN AS {seccode}_OPEN,
                {seccode}.HIGH AS {seccode}_HIGH,
                {seccode}.LOW AS {seccode}_LOW,
                {seccode}.CLOSE AS {seccode}_CLOSE,
                {seccode}.VOL AS {seccode}_VOL
            """
            condWHERE_SecIds = f"{seccode}.security_id = {security_id}"

            query = f"""
                SELECT {fieldsSELECT}
                FROM quote {seccode}
                WHERE {condWHERE_SecIds}
                ORDER BY {seccode}.DATE_TIME ASC
            """
            return query

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()

    def __buildQueryFromSecCodes(self, securities, since):
        date = since.strftime('%Y-%m-%d %H:%M:%S%z')
        log.info(f"Since: {date}")

        if len(securities) == 1:
            return self.__buildQueryFromOneSecCode(securities)

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()

            seccodes = [s['seccode'] for s in securities]
            securityIds = []
            listSelectFields = []

            for security in securities:
                query = """
                    SELECT id 
                    FROM security 
                    WHERE CODE = %s AND BOARD = %s
                """
                cursor.execute(query, (security['seccode'], security['board']))
                result = cursor.fetchone()

                if result is None:
                    raise RuntimeError(f"getDataFrames: {security['seccode']} not found")

                security_id = str(result[0])
                fields = f"""
                    {security['seccode']}.OPEN AS {security['seccode']}_OPEN,
                    {security['seccode']}.HIGH AS {security['seccode']}_HIGH,
                    {security['seccode']}.LOW AS {security['seccode']}_LOW,
                    {security['seccode']}.CLOSE AS {security['seccode']}_CLOSE,
                    {security['seccode']}.VOL AS {security['seccode']}_VOL
                """
                listSelectFields.append(fields)
                securityIds.append((security['seccode'], security_id))

            fieldsSELECT = f"{seccodes[0]}.DATE_TIME, " + ',\n'.join(listSelectFields)
            fieldsFROM = ', '.join([f"quote {s}" for s in seccodes])

            condWHERE_date = f"{seccodes[0]}.DATE_TIME > '{date}' AND "
            condWHERE_SecIds = ' AND '.join([f"{s}.security_id = {id}" for s, id in securityIds])

            query = f"""
                SELECT {fieldsSELECT}
                FROM {fieldsFROM}
                WHERE {condWHERE_date}{condWHERE_SecIds}
                ORDER BY {seccodes[0]}.DATE_TIME ASC
            """
            return query

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()

    def __buildQueryGetClosePrice(self, securities, since):
        date = since.strftime('%Y-%m-%d %H:%M:%S%z')
        log.info(f"Since: {date}")

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()

            seccodes = [s['seccode'] for s in securities]
            securityIds = []
            listSelectFields = []

            for security in securities:
                query = """
                    SELECT id 
                    FROM security 
                    WHERE CODE = %s AND BOARD = %s
                """
                cursor.execute(query, (security['seccode'], security['board']))
                result = cursor.fetchone()

                if result is None:
                    raise RuntimeError(f"GetClosePrice: {security['seccode']} not found")

                security_id = str(result[0])
                securityIds.append((security['seccode'], security_id))

            fields = f"""
                {securities[0]['seccode']}.CLOSE AS {securities[0]['seccode']}_CLOSE,
                {securities[0]['seccode']}.OPEN AS {securities[0]['seccode']}_OPEN,
                {securities[0]['seccode']}.HIGH AS {securities[0]['seccode']}_HIGH,
                {securities[0]['seccode']}.LOW AS {securities[0]['seccode']}_LOW,
                {securities[0]['seccode']}.VOL AS {securities[0]['seccode']}_VOL
            """
            listSelectFields.append(fields)

            for security in securities[1:]:
                fields = f"{security['seccode']}.CLOSE AS {security['seccode']}_CLOSE"
                listSelectFields.append(fields)

            fieldsSELECT = f"{seccodes[0]}.DATE_TIME AS DATE_TIME, " + ',\n'.join(listSelectFields)
            fieldsFROM = ', '.join([f"quote {s}" for s in seccodes])

            condWHERE_date = f"{seccodes[0]}.DATE_TIME > '{date}' AND "
            condWHERE_SecIds = ' AND '.join([f"{s}.security_id = {id}" for s, id in securityIds])

            query = f"""
                SELECT {fieldsSELECT}
                FROM {fieldsFROM}
                WHERE {condWHERE_date}{condWHERE_SecIds}
                ORDER BY {seccodes[0]}.DATE_TIME ASC
            """
            return query

        except psycopg2.Error as error:
            log.error("Failed to execute database operation", error)
        finally:
            if conn:
                conn.close()

    def retrieveData(self, securities, periods, since=None, target=None):
        query = self.__buildQueryGetClosePrice(securities, since)
        log.info("Reading from database...")
        fmt = '%Y-%m-%d %H:%M:%S%z'
        seccodes = [s['seccode'] for s in securities]
        agg_dict = {}
        name_dict = {}

        for seccode in seccodes:
            agg_dict[f"{seccode}_CLOSE"] = ['last']
            agg_dict[f"{seccode}_OPEN"] = ['first']
            agg_dict[f"{seccode}_HIGH"] = ['max']
            agg_dict[f"{seccode}_LOW"] = ['min']
            agg_dict[f"{seccode}_VOL"] = ['sum']

            name_dict[f"{seccode}_CLOSE"] = None
            name_dict[f"{seccode}_OPEN"] = None
            name_dict[f"{seccode}_HIGH"] = None
            name_dict[f"{seccode}_LOW"] = None
            name_dict[f"{seccode}_VOL"] = None

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            data = pd.read_sql_query(
                query,
                conn,
                parse_dates={'DATE_TIME': fmt},
                index_col='DATE_TIME'
            )

            for p in periods:
                log.info(f"Re-sampling to {p}...")

                df = data.resample(p).agg(agg_dict, names=name_dict).dropna()
                df.columns = df.columns.droplevel(1)
                self.__dataFrames[p] = df

            return self.__dataFrames

        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        finally:
            if conn:
                conn.close()
                log.info("Finished reading")

    def getSecurityInfo(self, security):
        errMsg = f"__getSecurityIdSQL: {security['seccode']} not found"

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            query = """
                SELECT DECIMALS, MARKET
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (security['seccode'], security['board']))
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError(errMsg)

            cursor.close()
            return result[0], result[1]

        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        except Exception as inst:
            log.error(str(inst.args))

    def getSecurityAlgParams(self, security):
        errMsg = f"{security['seccode']} not found"

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            query = """
                SELECT ALG_PARAMETERS
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (security['seccode'], security['board']))
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError(errMsg)

            cursor.close()
            params = result[0]  # No need to use json.loads()            
            return params

        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        except Exception as inst:
            log.error(str(inst.args))

    def syncData(self, data):
        log.info("Synchronizing database...")

        if not data:
            data.update(self.searchData(self.since))
            return

        self.since = self.since + dt.timedelta(minutes=1)
        self.until = self.since + dt.timedelta(minutes=self.numTestSample)

        if self.MODE == 'TRAIN_OFFLINE':
            data.update(self.searchData(self.since))

        elif self.MODE == 'TEST_OFFLINE':
            data.update(self.searchData(self.since, self.until))

        elif self.MODE in ['TEST_ONLINE', 'OPERATIONAL']:
            since = dt.datetime.now() - dt.timedelta(days=5)
            while True:
                try:
                    dfs = self.searchData(since)
                    if dfs:
                        log.info("Data found for synchronization.")
                    else:
                        log.error("No data returned for synchronization.")
                except Exception as e:
                    log.error(f"Error during synchronization: {e}")  
                if not self.isSufficientData(dfs):
                    continue
                if self.isPeriodSynced(dfs):
                    break
                time.sleep(1.5)
            for p in self.periods:
                data[p] = pd.concat([data[p], dfs[p]]).drop_duplicates().sort_index()


    def searchData(self, since, until=None):
        securities = self.securities
        periods = self.periods
        log.info(f"Searching into database..{since} til {until}")
        tb, te = self.between_time
        agg_dict = {
            'minprice': 'min',
            'maxprice': 'max',
            'startprice': 'first',
            'endprice': 'last',
            'hastrade': 'last',
            'addedvolume': 'sum',
            'numberoftrades': 'sum'
        }    
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            data = {}
            for p in periods:
                list_df = []
                for sec in securities:
                    query = self.__querySearchSec(sec, since, until)
                    df = pd.read_sql_query(query, conn)
                    # Convert 'date_time' to datetime with UTC
                    df['date_time'] = pd.to_datetime(df['date_time'], utc=True)
    
                    # Set 'date_time' as the index
                    df.set_index('date_time', inplace=True)                    
    
                    # Ensure 'since' is timezone aware and in UTC
                    if since.tzinfo is None:
                        since = since.replace(tzinfo=pytz.UTC)
    
                    df = df.loc[since:]
                    df = df.between_time(tb, te)
    
                    # Resampling
                    resampled_df = df.resample(p, closed='right', origin='start_day', convention='end').agg(agg_dict).dropna()
    
                    # Adding back the non-numeric column 'mnemonic' after resampling
                    resampled_df['mnemonic'] = sec['seccode']
    
                    list_df.append(resampled_df)
    
                df = pd.concat(list_df, axis=0)
                if df.empty:
                    logging.error(f"No data found for seccode: {sec}")
                    return None  
                data[p] = df
    
            return data
    
        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        finally:
            if conn:
                conn.close()
                log.debug("Finished searching")
    

    def isSufficientData(self, dataFrame):
        dataFrame = dataFrame['1Min']
        msg = 'There are only %s samples now; you need at least %s samples for the model to predict.'
        sufficient = True
        for sec in self.securities:
            seccode = sec['seccode']
            minNumPastSamples = sec['params']['minNumPastSamples']
            df = dataFrame[dataFrame['mnemonic'] == seccode]
            numSamplesNow = len(df.index)
            if numSamplesNow < minNumPastSamples:
                logging.warning(msg, numSamplesNow, minNumPastSamples)
                sufficient = False
                break

        return sufficient

    def isPeriodSynced(self, dfs):
        synced = False
        numPeriod = 1
        dataFrame = dfs['1Min']
        dataFrame_1min = dfs['1Min']

        for sec in self.securities:
            seccode = sec['seccode']
            df = dataFrame[dataFrame['mnemonic'] == seccode]
            df_1min = dataFrame_1min[dataFrame_1min['mnemonic'] == seccode]
            timelastPeriod = df.tail(1).index
            timelastPeriod = timelastPeriod.to_pydatetime()
            timelast1Min = df_1min.tail(1).index
            timelast1Min = timelast1Min.to_pydatetime()
            nMin = -numPeriod + 1
            timeAux = timelast1Min + dt.timedelta(minutes=nMin)

            if timeAux >= timelastPeriod and self.lastUpdate != timelastPeriod:
                synced = True
                self.lastUpdate = timelastPeriod

            logging.debug(f'TimelastPeriod: {timelastPeriod}')
            logging.debug(f'Timelast1Min: {timelast1Min}')
            logging.debug(f'TimeAux: {timeAux}')
            logging.debug(f'Period synced: {synced}')

        return synced

    def __querySearchSec(self, security, since, until):
        
        date = since.strftime('%Y-%m-%d %H:%M:%S%z')
        untilDate = ""
        if until is not None:
            untilDate = until.strftime('%Y-%m-%d %H:%M:%S%z')
        log.debug(f"Since: {date}")
        log.debug(f"Until: {untilDate}" if untilDate else "Until: None")
    
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
    
            query = """
                SELECT id 
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (security['seccode'], security['board']))
            result = cursor.fetchone()
    
            if result is None:
                raise RuntimeError(f"{security['seccode']} not found")
    
            security_id = str(result[0])
    
            fieldsSELECT = f"""
                '{security['seccode']}' AS mnemonic,
                low AS minprice,
                high AS maxprice,
                open AS startprice,
                close AS endprice,
                1 AS hastrade,
                vol AS addedvolume,
                vol AS numberoftrades,
                date_time AS date_time
            """
    
            fieldsFROM = "quote"
    
            condWHERE = f"security_id = {security_id} AND date_time > '{date}'"
            if until is not None:
                condWHERE += f" AND date_time < '{untilDate}'"
    
            query = f"""
                SELECT {fieldsSELECT}
                FROM {fieldsFROM}
                WHERE {condWHERE}
                ORDER BY date_time ASC
            """
            return query
    
        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        finally:
            if conn:
                conn.close()


    def __getSecurityIdSQL(self, board, seccode):
        
        errMsg = f"__getSecurityIdSQL: seccode: {seccode} not found"
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            query = """
                SELECT id 
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (seccode, board))
            result = cursor.fetchone()

            if result is None:
                raise RuntimeError(errMsg)

            cursor.close()
            return str(result[0])

        except psycopg2.Error as error:
            log.error(errMsg, error)


    def storeCandles(self, historyCandlePacket):
        
        board = historyCandlePacket.board
        seccode = historyCandlePacket.seccode
        security_id = self.__getSecurityIdSQL(board, seccode)

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            cursor.execute("BEGIN")

            for c in historyCandlePacket.items:
                if not self.isConsolidatedCandle(c):
                    continue

                values = (
                    c.date,  # No need for datetime2SQLString, store directly as TIMESTAMPTZ
                    c.open,
                    c.high,
                    c.low,
                    c.close,
                    c.volume,
                    security_id
                )

                query_insert = """
                    INSERT INTO quote
                    (DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, security_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (DATE_TIME, security_id) 
                    DO UPDATE SET 
                        OPEN = EXCLUDED.OPEN,
                        HIGH = EXCLUDED.HIGH,
                        LOW = EXCLUDED.LOW,
                        CLOSE = EXCLUDED.CLOSE,
                        VOL = EXCLUDED.VOL;
                """

                cursor.execute(query_insert, values)

            cursor.execute("COMMIT")
            conn.commit()
            cursor.close()

        except Exception as inst:
            errMsg = f"{type(inst)}\n{inst.args}\n{inst}"
            log.error(errMsg)
            log.error("Failed to commit")

        finally:
            if conn:
                conn.close()

    def isConsolidatedCandle(self, candle):
        
        moscowTimeZone = pytz.timezone('Europe/Moscow')
        moscowDateTime = dt.datetime.now(moscowTimeZone)
        moscowDateTime_minus1min = moscowDateTime - dt.timedelta(minutes=1)

        if moscowDateTime_minus1min > candle.date:
            log.debug("Yes, it goes to the archive")
            return True
        else:
            log.debug("No, it isn't consolidated yet, not a minute old yet")
            return False

    def __takeDataFromAnotherSec(self):
        new_security_id = 7
        previous_security_id = 4

        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()

            query = f"""
                INSERT INTO quote (DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, security_id)
                SELECT DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, {new_security_id}
                FROM quote
                WHERE security_id = {previous_security_id}
                ON CONFLICT (DATE_TIME, security_id)
                DO NOTHING;
            """

            cursor.execute(query)
            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to insert data", error)

        finally:
            if conn:
                conn.close()

    
                
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
            s.connect((cm.transaqConnectorHost,cm.transaqConnectorPort))    
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
    
    def resampleDataFrames(self, securities, periods=['1Min'], since=None, target=None):
        try:
            query = self.__buildQueryFromSecCodes(securities, since)
            seccodes = [s['seccode'] for s in securities]
            agg_dict = {}
            name_dict = {}
            for seccode in seccodes:
                agg_dict[f"{seccode}_OPEN"] = ['first']
                agg_dict[f"{seccode}_HIGH"] = ['max']
                agg_dict[f"{seccode}_LOW"] = ['min']
                agg_dict[f"{seccode}_CLOSE"] = ['last']
                agg_dict[f"{seccode}_VOL"] = ['sum']
    
                name_dict[f"{seccode}_OPEN"] = None
                name_dict[f"{seccode}_HIGH"] = None
                name_dict[f"{seccode}_LOW"] = None
                name_dict[f"{seccode}_CLOSE"] = None
                name_dict[f"{seccode}_VOL"] = None
    
            log.info("Reading from database...")
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            data = pd.read_sql_query(
                query,
                conn,
                index_col='DATE_TIME'
            )
    
            for period in periods:
                log.info(f"Re-sampling to {period}...")
    
                self.__dataFrames[period] = data.resample(period).agg(agg_dict, names=name_dict).dropna()
    
            return self.__dataFrames
    
        except psycopg2.Error as error:
            log.error("Failed to read from database", error)
        finally:
            if conn:
                conn.close()
                log.info("Finished")

    
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
            moscowDateTime = dt.datetime.now(moscowTimeZone)
            stringMoscowMinute =  str(moscowDateTime.year).zfill(4) 
            stringMoscowMinute += str(moscowDateTime.month).zfill(2) 
            stringMoscowMinute += str(moscowDateTime.day).zfill(2) + " " 
            stringMoscowMinute += str(moscowDateTime.hour).zfill(2) 
            stringMoscowMinute += str(moscowDateTime.minute).zfill(2) + "00"
            
            shouldGo = True if ( stringMoscowMinute >  values[0]) else False
            
            if shouldGo == True: log.info( infoMsg )
                
        return shouldGo
    

    def __parse_XML2sql(self, xml_string, mode):
        
        try:
            xtree = ElementTree(fromstring(xml_string))
            xroot = xtree.getroot()
            if xroot.attrib.get('board') is not None:
                board = xroot.attrib.get('board')
            if xroot.attrib.get('seccode') is not None:
                seccode = xroot.attrib.get('seccode')
    
            security_id = self.__getSecurityIdSQL(board, seccode)
    
            log.info("Writing to database...")
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            cursor.execute("BEGIN")
    
            for candle in xroot.findall('candle'):
                values = self.__getCandleValues(candle, security_id)
    
                if not self.__shouldGo2Archive(mode, values):
                    continue
    
                query_insert = """
                    INSERT INTO quote 
                    (DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, security_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (DATE_TIME, security_id) 
                    DO UPDATE SET 
                        OPEN = EXCLUDED.OPEN,
                        HIGH = EXCLUDED.HIGH,
                        LOW = EXCLUDED.LOW,
                        CLOSE = EXCLUDED.CLOSE,
                        VOL = EXCLUDED.VOL;
                """
                cursor.execute(query_insert, values)
    
            cursor.execute("COMMIT")
            conn.commit()
            cursor.close()
    
        except Exception as inst:
            log.error(f"Failed to parse XML and write to SQL: {inst}")
        finally:
            if conn:
                conn.close()
    
    

    def datetime2SQLString(self, datetime):
        
        stringDatetime =  str(datetime.year).zfill(4) 
        stringDatetime += str(datetime.month).zfill(2) 
        stringDatetime += str(datetime.day).zfill(2) + " " 
        stringDatetime += str(datetime.hour).zfill(2) 
        stringDatetime += str(datetime.minute).zfill(2) + "00"
        return stringDatetime
            
    
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
                self.__sendHistoryRequest(security['board'],security['seccode'],'1', count, reset )                
                lastCandle = False               
                while not lastCandle:
                    data = ''
                    time.sleep(2)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((cm.transaqConnectorHost,cm.transaqConnectorPort))
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
        log.info("Having security_id: " + str(security_id))
        print("Having security_id: " + str(security_id))
    
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
    
            with open(fileName, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
    
                query_insert = """
                    INSERT INTO quote 
                    (DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, security_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (DATE_TIME, security_id) 
                    DO UPDATE SET 
                        OPEN = EXCLUDED.OPEN,
                        HIGH = EXCLUDED.HIGH,
                        LOW = EXCLUDED.LOW,
                        CLOSE = EXCLUDED.CLOSE,
                        VOL = EXCLUDED.VOL;
                """
    
                for data in reader:
                    data = data[2:]  # Assuming this only works for a specific DB format
                    data[0:2] = [' '.join(data[0:2])]  # Join date and time
                    data.append(security_id)
                    cursor.execute(query_insert, data)
    
            conn.commit()
            log.info("Finished inserting CSV data into the database.")
            print("Finished inserting CSV data into the database.")
    
        except psycopg2.Error as error:
            log.error("Failed to insert CSV data", error)
        except Exception as inst:
            log.error(str(inst))
        finally:
            if conn:
                conn.close()
    
    
    def getPlatformDetails(self, securities):
        
        security = securities[0]
        errMsg = f"{security['seccode']} not found"
    
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
    
            query = """
                SELECT platform
                FROM security 
                WHERE CODE = %s AND BOARD = %s
            """
            cursor.execute(query, (security['seccode'], security['board']))
            result = cursor.fetchone()
    
            if result is None:
                raise RuntimeError(errMsg)
    
            conn.close()
    
            platform_details = json.loads(result[0])
            return platform_details
    
        except psycopg2.Error as error:
            log.error("Failed to read platform details", error)
        except Exception as inst:
            log.error(str(inst))
        finally:
            if conn:
                conn.close()
                
                
    def store_candles(self, candles, security):
        try:
            seccode = security['seccode']
            board = security['board']
            
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            cursor.execute("BEGIN")
            
            security_id = self.__getSecurityIdSQL(board, seccode)
    
            for index, row in candles.iterrows():
                values = (
                    index,  # Use the index as the timestamp
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    security_id
                )
    
                query_insert = """
                    INSERT INTO quote
                    (DATE_TIME, OPEN, HIGH, LOW, CLOSE, VOL, security_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (DATE_TIME, security_id) 
                    DO UPDATE SET 
                        OPEN = EXCLUDED.OPEN,
                        HIGH = EXCLUDED.HIGH,
                        LOW = EXCLUDED.LOW,
                        CLOSE = EXCLUDED.CLOSE,
                        VOL = EXCLUDED.VOL;
                """
    
                cursor.execute(query_insert, values)
    
            cursor.execute("COMMIT")
            conn.commit()
            cursor.close()
    
        except Exception as e:
            log.error("Failed to commit: %s", e)
        finally:
            if conn:
                conn.close()

    def insert_alpaca_tickers(self, json_file_path):
        conn = None  # Initialize conn to None to avoid UnboundLocalError
        
        try:
            # Check if the JSON file exists
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"No such file or directory: '{json_file_path}'")
    
            # Load the JSON data from the file
            with open(json_file_path, 'r') as json_file:
                tickers = json.load(json_file)
            
            # Prepare the common data fields
            period = 1
            board = "EQTY"
            decimals = 3
            market = "NASDAQ"
            alg_parameters = {
                "algorithm": "peaks_and_valleys",
                "entryByMarket": False,
                "entryTimeSeconds": 180,
                "positionQuantity": 1,
                "minNumPastSamples": 51,
                "longPositionMargin": 10,
                "shortPositionMargin": 10,
                "stopLossCoefficient": 6,
                "acceptableTrainingError": 0.000192
            }
            platform = {
                "name": "alpaca",
                "secrets": {
                    "api_key": "AK2FP97N6GXE19GZ3FGK",
                    "api_secret": "5oJkbjdAXoeYn4aKLKLn2dgSY8AUFRC4no2hhKYc",
                    "endpoint": "https://api.alpaca.markets"
                }
            }
            
            # Establish database connection
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
        
            # Iterate over the tickers and insert the valid ones
            for ticker in tickers:
                alpaca_ticker = ticker.get('AlpacaTicker')
                
                if alpaca_ticker:  # Only process entries with a valid AlpacaTicker
                    query_insert = """
                        INSERT INTO security
                        (code, period, board, decimals, market, alg_parameters, platform)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (code) DO NOTHING;
                    """
        
                    cursor.execute(query_insert, (
                        alpaca_ticker,
                        period,
                        board,
                        decimals,
                        market,
                        json.dumps(alg_parameters),  # Convert to JSONB format
                        json.dumps(platform)         # Convert to JSONB format
                    ))
        
            # Commit the changes
            conn.commit()
            cursor.close()
        
        except Exception as e:
            log.error("Failed to insert Alpaca tickers: %s", e)
        
        finally:
            if conn:
                conn.close()
    
    
    def store_bar(self, seccode, bar_data):
        """
        Store the incoming quote data into the database or update in-memory storage.
        """
        try:
            conn = psycopg2.connect(dbname=cm.dbname, user=cm.user, password=cm.password, host=cm.host)
            cursor = conn.cursor()
            log.info(f"Storing quote for {seccode}...")

            query = """
                INSERT INTO quote (date_time, open, high, low, close, vol, security_id)
                VALUES (%s, %s, %s, %s, %s, %s, 
                (SELECT id FROM security WHERE code = %s LIMIT 1))
                ON CONFLICT (date_time, security_id) DO UPDATE 
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    vol = EXCLUDED.vol;
            """
            cursor.execute(query, (
                bar_data['timestamp'], 
                bar_data['open'], 
                bar_data['high'], 
                bar_data['low'], 
                bar_data['close'], 
                bar_data['volume'], 
                seccode
            ))

            conn.commit()
            cursor.close()

        except psycopg2.Error as error:
            log.error("Failed to store quote data", error)
        finally:
            if conn:
                conn.close()
                log.info("Database connection closed")    
   



if __name__ == "__main__":
    
    ds = DataServer()
    filePath = "../TradingPlatforms/Alpaca/AlpacaTickers.json"
    ds.insert_alpaca_tickers(filePath)
     
#     security_id = 1
#     fileName = './bkp/GAZR/GAZR.txt'
#     connector.insertCSV2DB(security_id, fileName)
    
#     security_id = 2    
#     fileName = './bkp/SPFB.SBRF/SPFB.SBRF-9.20_190101_200726.csv'
#     connector.insertCSV2DB(security_id, fileName)
    
#    connector.updateDBquotes(self.securities, 'archive')        
#    period = '2Min'
#
#    while True:            
#        connector.updateDBquotes(securities, 'online')
#        connector.resampleDataFrames(securities, period )
#        dataFrames = connector.getDataFrames()
#        log.info(dataFrames[period].tail())
   

    