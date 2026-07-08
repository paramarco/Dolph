#!/usr/bin/env python3
"""IB Market Scanner — uses Interactive Brokers API to find stocks with
unusual activity (volume surges, price moves, 52-week highs).

These pre-filtered stocks are then deep-analyzed with yfinance for
Three Power Moves detection.

Requires IB Gateway running on the configured port.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

log = logging.getLogger("ScanEngine")


def scan_ib_hot_stocks(host='127.0.0.1', port=4001, client_id=98):
    """Connect to IB and run multiple scanner queries to find hot stocks.

    Returns list of unique stock dicts ready for deep analysis.
    """
    try:
        sys.path.insert(0, '/opt/venv/lib/python3.11/site-packages')
        from ib_insync import IB, ScannerSubscription
    except ImportError:
        log.error("ib_insync not available")
        return []

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
        log.info(f"IB connected on port {port}, accounts: {ib.managedAccounts()}")
    except Exception as e:
        log.warning(f"IB connection failed (port {port}): {e}")
        return []

    # Define scans: (scan_code, location, max_rows)
    scan_configs = [
        # US Major exchanges (NYSE, NASDAQ large/mid cap)
        ('TOP_VOLUME_RATE',  'STK.US.MAJOR', 50),
        ('HOT_BY_VOLUME',    'STK.US.MAJOR', 50),
        ('TOP_PERC_GAIN',    'STK.US.MAJOR', 50),
        ('HIGH_VS_52W_HL',   'STK.US.MAJOR', 50),

        # US Minor exchanges (OTC, small/micro caps)
        ('TOP_VOLUME_RATE',  'STK.US.MINOR', 50),
        ('HOT_BY_VOLUME',    'STK.US.MINOR', 50),
        ('TOP_PERC_GAIN',    'STK.US.MINOR', 30),

        # European exchanges (when market is open)
        ('TOP_VOLUME_RATE',  'STK.EU',       50),
        ('HOT_BY_VOLUME',    'STK.EU',       50),
        ('TOP_PERC_GAIN',    'STK.EU',       30),
        ('HIGH_VS_52W_HL',   'STK.EU',       50),
    ]

    all_symbols = {}  # symbol → {name, exchange, ...}

    for scan_code, location, max_rows in scan_configs:
        try:
            sub = ScannerSubscription(
                instrument='STK',
                locationCode=location,
                scanCode=scan_code,
                numberOfRows=max_rows,
                stockTypeFilter='STOCK',
            )
            results = ib.reqScannerData(sub, [])

            for r in results:
                cd = r.contractDetails
                symbol = cd.contract.symbol
                if symbol not in all_symbols:
                    # Map to yfinance ticker for EU stocks
                    yf_symbol = _ib_to_yfinance(cd)
                    all_symbols[symbol] = {
                        'symbol': yf_symbol,
                        'name': cd.longName or '',
                        'sector': '',
                        'industry': cd.industry or '',
                        'exchange': cd.contract.primaryExchange or location,
                        'currency': cd.contract.currency or 'USD',
                        'ib_scan': scan_code,
                    }

            count = len(results)
            if count > 0:
                log.info(f"IB scan {scan_code} @ {location}: {count} results")

        except Exception as e:
            log.debug(f"IB scan {scan_code} @ {location} failed: {e}")

    ib.disconnect()

    stocks = list(all_symbols.values())
    log.info(f"IB scanner total: {len(stocks)} unique hot stocks")
    return stocks


def _ib_to_yfinance(contract_details):
    """Convert IB contract to yfinance ticker symbol."""
    cd = contract_details
    symbol = cd.contract.symbol
    exchange = cd.contract.primaryExchange or ''
    currency = cd.contract.currency or 'USD'

    # US stocks: symbol as-is
    if currency == 'USD':
        return symbol

    # EU stocks: add exchange suffix for yfinance
    suffix_map = {
        'IBIS': '.DE', 'XETRA': '.DE',
        'SBF': '.PA', 'ENEXT.BE': '.BR',
        'LSE': '.L', 'LSEETF': '.L',
        'BM': '.MC', 'SIBE': '.MC',
        'BVME': '.MI', 'BVME.ETF': '.MI',
        'AEB': '.AS', 'ENEXT.BE': '.BR',
        'SWX': '.SW', 'EBS': '.SW',
        'OMX': '.ST', 'OMXHEX': '.HE',
        'OMXCOP': '.CO', 'OB': '.OL',
        'VSE': '.VI',
    }

    suffix = suffix_map.get(exchange, '')
    if suffix:
        return f"{symbol}{suffix}"

    # Fallback: try currency
    currency_suffix = {
        'EUR': '.DE',  # default to XETRA
        'GBP': '.L',
        'CHF': '.SW',
        'SEK': '.ST',
        'DKK': '.CO',
        'NOK': '.OL',
    }
    suffix = currency_suffix.get(currency, '')
    return f"{symbol}{suffix}" if suffix else symbol


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    stocks = scan_ib_hot_stocks()
    print(f"\n{len(stocks)} hot stocks found via IB scanner:")
    for s in stocks[:20]:
        print(f"  {s['symbol']:12s} {s['name'][:40]:40s} [{s['ib_scan']}]")
