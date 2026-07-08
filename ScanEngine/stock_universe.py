#!/usr/bin/env python3
"""Fetch stock universe for scanning: S&P 500 + STOXX Europe 600.

Prioritizes sectors in this order:
  1. Consumer Discretionary (brands like Ralph Lauren)
  2. Financials (fintech, financial services)
  3. Consumer Staples (non-tech)
  4. Information Technology (semiconductors, AI)
  5. Industrials (aerospace & defense)
  6. All remaining sectors
"""
import logging
import requests
import pandas as pd
import io

log = logging.getLogger("ScanEngine")

# Sector priority for scanning order
SECTOR_PRIORITY = [
    'Consumer Discretionary',
    'Financials',
    'Consumer Staples',
    'Information Technology',
    'Industrials',
    'Health Care',
    'Communication Services',
    'Energy',
    'Materials',
    'Real Estate',
    'Utilities',
]

HEADERS = {'User-Agent': 'Mozilla/5.0 (DolphScanner/1.0)'}


def fetch_sp500():
    """Fetch S&P 500 components from Wikipedia."""
    try:
        r = requests.get(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            headers=HEADERS, timeout=30)
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        stocks = []
        for _, row in df.iterrows():
            symbol = str(row['Symbol']).replace('.', '-')  # BRK.B → BRK-B for yfinance
            stocks.append({
                'symbol': symbol,
                'name': row.get('Security', ''),
                'sector': row.get('GICS Sector', ''),
                'industry': row.get('GICS Sub-Industry', ''),
                'exchange': 'US',
                'currency': 'USD',
            })
        log.info(f"Fetched {len(stocks)} S&P 500 stocks")
        return stocks
    except Exception as e:
        log.error(f"Failed to fetch S&P 500: {e}")
        return []


def fetch_stoxx600():
    """Fetch STOXX Europe 600 from Wikipedia."""
    try:
        r = requests.get(
            'https://en.wikipedia.org/wiki/STOXX_Europe_600',
            headers=HEADERS, timeout=30)
        tables = pd.read_html(io.StringIO(r.text))
        # Find the table with stock data (largest table)
        df = max(tables, key=len)
        stocks = []
        for _, row in df.iterrows():
            # STOXX 600 Wikipedia table varies; try common column names
            symbol = str(row.get('Ticker', row.get('Symbol', row.iloc[0])))
            name = str(row.get('Company', row.get('Name', row.iloc[1] if len(row) > 1 else '')))
            sector = str(row.get('ICB Sector', row.get('Sector', '')))
            country = str(row.get('Country', ''))

            # Map to yfinance ticker (add exchange suffix)
            yf_ticker = _map_eu_ticker(symbol, country)
            if yf_ticker:
                stocks.append({
                    'symbol': yf_ticker,
                    'name': name,
                    'sector': sector,
                    'industry': '',
                    'exchange': 'EU',
                    'currency': 'EUR',
                })
        log.info(f"Fetched {len(stocks)} STOXX 600 stocks")
        return stocks
    except Exception as e:
        log.error(f"Failed to fetch STOXX 600: {e}")
        return []


def _map_eu_ticker(symbol, country):
    """Map European ticker to yfinance format (e.g., SAP → SAP.DE)."""
    country = country.strip().lower() if country else ''
    suffixes = {
        'germany': '.DE', 'deutschland': '.DE',
        'france': '.PA',
        'united kingdom': '.L', 'uk': '.L',
        'spain': '.MC',
        'italy': '.MI',
        'netherlands': '.AS',
        'switzerland': '.SW',
        'sweden': '.ST',
        'denmark': '.CO',
        'norway': '.OL',
        'finland': '.HE',
        'belgium': '.BR',
        'portugal': '.LS',
        'austria': '.VI',
        'ireland': '.IR',
    }
    suffix = suffixes.get(country, '')
    if not suffix:
        return None  # Skip if we can't determine the exchange
    return f"{symbol}{suffix}"


def get_scan_universe():
    """Get the full stock universe for scanning, ordered by sector priority."""
    sp500 = fetch_sp500()
    stoxx600 = fetch_stoxx600()

    all_stocks = sp500 + stoxx600

    # Sort by sector priority
    def sector_order(stock):
        sector = stock.get('sector', '')
        try:
            return SECTOR_PRIORITY.index(sector)
        except ValueError:
            return len(SECTOR_PRIORITY)  # Unknown sectors last

    all_stocks.sort(key=sector_order)

    log.info(f"Total scan universe: {len(all_stocks)} stocks "
             f"(US: {len(sp500)}, EU: {len(stoxx600)})")
    return all_stocks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    stocks = get_scan_universe()
    print(f"\nTotal: {len(stocks)} stocks")
    sectors = {}
    for s in stocks:
        sec = s['sector']
        sectors[sec] = sectors.get(sec, 0) + 1
    for sec, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"  {sec}: {count}")


def fetch_us_full():
    """Fetch ALL US stocks from NASDAQ API (~7,000 stocks)."""
    import requests as _req
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    stocks = []
    for exchange in ['NASDAQ', 'NYSE', 'AMEX']:
        try:
            r = _req.get(
                f'https://api.nasdaq.com/api/screener/stocks?tableType=main&exchange={exchange}&limit=10000&offset=0',
                headers=headers, timeout=30)
            data = r.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])
            for row in rows:
                symbol = row.get('symbol', '').strip()
                name = row.get('name', '')
                if symbol and '^' not in symbol and '/' not in symbol:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'sector': '',
                        'industry': '',
                        'exchange': exchange,
                        'currency': 'USD',
                    })
        except Exception as e:
            log.error(f"Failed to fetch {exchange}: {e}")
    log.info(f"Fetched {len(stocks)} US stocks (NASDAQ+NYSE+AMEX)")
    return stocks


def get_full_scan_universe():
    """Get the FULL stock universe: all US + STOXX 600 EU.
    
    ~7,000 US + ~460 EU = ~7,500 total.
    For EU, we still use STOXX 600 (reliable yfinance tickers).
    """
    us_full = fetch_us_full()
    stoxx600 = fetch_stoxx600()

    all_stocks = us_full + stoxx600
    log.info(f"Full scan universe: {len(all_stocks)} stocks "
             f"(US: {len(us_full)}, EU: {len(stoxx600)})")
    return all_stocks


def fetch_eu_indices():
    """Fetch EU stocks from major national indices via Wikipedia.
    
    Covers ~1,100 stocks from 18 indices across UK, Germany, France,
    Spain, Italy, Switzerland, Netherlands, Nordics, Belgium, Portugal, Austria.
    """
    indices = {
        'FTSE 100': ('https://en.wikipedia.org/wiki/FTSE_100_Index', '.L', 'GBP'),
        'FTSE 250': ('https://en.wikipedia.org/wiki/FTSE_250_Index', '.L', 'GBP'),
        'DAX': ('https://en.wikipedia.org/wiki/DAX', '.DE', 'EUR'),
        'MDAX': ('https://en.wikipedia.org/wiki/MDAX', '.DE', 'EUR'),
        'SDAX': ('https://en.wikipedia.org/wiki/SDAX', '.DE', 'EUR'),
        'CAC 40': ('https://en.wikipedia.org/wiki/CAC_40', '.PA', 'EUR'),
        'CAC Next 20': ('https://en.wikipedia.org/wiki/CAC_Next_20', '.PA', 'EUR'),
        'IBEX 35': ('https://en.wikipedia.org/wiki/IBEX_35', '.MC', 'EUR'),
        'FTSE MIB': ('https://en.wikipedia.org/wiki/FTSE_MIB', '.MI', 'EUR'),
        'SMI': ('https://en.wikipedia.org/wiki/Swiss_Market_Index', '.SW', 'CHF'),
        'AEX': ('https://en.wikipedia.org/wiki/AEX_index', '.AS', 'EUR'),
        'OMX 30': ('https://en.wikipedia.org/wiki/OMX_Stockholm_30', '.ST', 'SEK'),
        'OMX Copenhagen 25': ('https://en.wikipedia.org/wiki/OMX_Copenhagen_25', '.CO', 'DKK'),
        'OMX Helsinki 25': ('https://en.wikipedia.org/wiki/OMX_Helsinki_25', '.HE', 'EUR'),
        'OBX': ('https://en.wikipedia.org/wiki/OBX_Index', '.OL', 'NOK'),
        'BEL 20': ('https://en.wikipedia.org/wiki/BEL_20', '.BR', 'EUR'),
        'PSI 20': ('https://en.wikipedia.org/wiki/PSI-20', '.LS', 'EUR'),
        'ATX': ('https://en.wikipedia.org/wiki/Austrian_Traded_Index', '.VI', 'EUR'),
    }

    seen = set()
    stocks = []
    
    for idx_name, (url, suffix, currency) in indices.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            tables = pd.read_html(io.StringIO(r.text))
            df = max(tables, key=len)
            
            # Find ticker/symbol column
            ticker_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ['ticker', 'symbol', 'code', 'epic']):
                    ticker_col = col
                    break
            
            # Find company name column
            name_col = None
            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ['company', 'name', 'constituent']):
                    name_col = col
                    break
            
            count = 0
            for _, row in df.iterrows():
                ticker = str(row[ticker_col]).strip() if ticker_col else ''
                name = str(row[name_col]).strip() if name_col else str(row.iloc[0])
                
                if not ticker or ticker == 'nan':
                    continue
                
                # Clean ticker and add suffix
                ticker = ticker.split('.')[0].split(':')[-1].strip()
                yf_ticker = f"{ticker}{suffix}"
                
                if yf_ticker in seen:
                    continue
                seen.add(yf_ticker)
                
                stocks.append({
                    'symbol': yf_ticker,
                    'name': name[:50],
                    'sector': '',
                    'industry': '',
                    'exchange': f'EU_{idx_name}',
                    'currency': currency,
                })
                count += 1
            
            if count > 0:
                log.info(f"  {idx_name}: {count} stocks")
                
        except Exception as e:
            log.debug(f"Failed to fetch {idx_name}: {e}")
    
    log.info(f"Fetched {len(stocks)} EU stocks from {len(indices)} national indices")
    return stocks


def get_full_scan_universe_v2():
    """Get the FULL stock universe: all US + EU national indices.
    
    ~7,000 US + ~1,100 EU = ~8,100 total.
    Uses EU national indices instead of just STOXX 600.
    """
    us_full = fetch_us_full()
    eu_indices = fetch_eu_indices()
    stoxx600 = fetch_stoxx600()
    
    # Merge EU: combine STOXX 600 with national indices, deduplicate
    eu_seen = set()
    eu_all = []
    for s in eu_indices + stoxx600:
        if s['symbol'] not in eu_seen:
            eu_seen.add(s['symbol'])
            eu_all.append(s)
    
    all_stocks = us_full + eu_all
    log.info(f"Full scan universe v2: {len(all_stocks)} stocks "
             f"(US: {len(us_full)}, EU: {len(eu_all)})")
    return all_stocks
