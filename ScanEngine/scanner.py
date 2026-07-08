#!/usr/bin/env python3
"""SCAN_ONLINE market scanner — orchestrates Three Power Moves detection
across US and EU stock markets.

Usage:
    python3 scanner.py                        # Full scan with email alerts
    python3 scanner.py --no-email             # Scan without sending email
    python3 scanner.py --symbol AAPL          # Scan single stock
    python3 scanner.py --sector Technology    # Scan specific sector
    python3 scanner.py --min-power-moves 2    # Only show 2+ power moves
"""
import sys
import os
import json
import logging
import datetime as dt
import psycopg2

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from ScanEngine.stock_universe import get_scan_universe, get_full_scan_universe, get_full_scan_universe_v2
from ScanEngine.ib_scanner import scan_ib_hot_stocks
from ScanEngine.power_moves import analyze_stock, screen_high_dividend_undervalued
from ScanEngine.alert_email import send_opportunity_alert, _format_dividend_section
from ScanEngine.portfolio_monitor import get_open_positions, check_positions_vs_warnings, compute_trailing_stops, format_portfolio_section

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
log = logging.getLogger("ScanEngine")

DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
          password='dolph_password', dbname='dolph_db', sslmode='disable',
          options='-c client_encoding=UTF8')


def parse_args():
    args = {
        'no_email': False,
        'symbol': None,
        'sector': None,
        'min_power_moves': 1,
        'full_market': False,
        'use_ib': False,
        'ib_port': 4001,
    }
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--no-email':
            args['no_email'] = True
        if arg == '--symbol' and i < len(sys.argv) - 1:
            args['symbol'] = sys.argv[i + 1]
        if arg == '--sector' and i < len(sys.argv) - 1:
            args['sector'] = sys.argv[i + 1]
        if arg == '--ib':
            args['use_ib'] = True
        if arg == '--ib-port' and i < len(sys.argv) - 1:
            args['ib_port'] = int(sys.argv[i + 1])
        if arg == '--full':
            args['full_market'] = True
        if arg == '--min-power-moves' and i < len(sys.argv) - 1:
            args['min_power_moves'] = int(sys.argv[i + 1])
    return args


def store_result(result):
    """Store scan result in scan_opportunities table."""
    try:
        # Convert numpy types to Python native (psycopg2 can't adapt numpy.bool_)
        def _py(v):
            if hasattr(v, 'item'):
                return v.item()
            return v

        conn = psycopg2.connect(**DB)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO scan_opportunities (
                scan_ts, symbol, company_name, sector, industry, exchange, currency,
                close_price, daily_change_pct,
                volume_today, volume_avg_50d, volume_ratio, volume_direction, volume_spike,
                resistance_level, support_level, consolidation_months, resistance_touches,
                breakout_detected,
                eps_latest, eps_record, eps_turned_positive, eps_trend, eps_quarters,
                power_moves_count, alert_type,
                daily_volatility_pct, suggested_stop_pct,
                analysis_notes
            ) VALUES (
                NOW(), %s, %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s
            )
        """, (
            _py(result['symbol']), _py(result.get('company_name')), _py(result.get('sector')),
            _py(result.get('industry')), _py(result.get('exchange', 'US')), _py(result.get('currency', 'USD')),
            _py(result.get('close_price')), _py(result.get('daily_change_pct')),
            _py(result.get('volume_today')), _py(result.get('volume_avg_50d')),
            _py(result.get('volume_ratio')), _py(result.get('volume_direction')), _py(result.get('volume_spike')),
            _py(result.get('resistance_level')), _py(result.get('support_level')),
            _py(result.get('consolidation_months')), _py(result.get('resistance_touches')),
            _py(result.get('breakout_detected')),
            _py(result.get('eps_latest')), _py(result.get('eps_record')), _py(result.get('eps_turned_positive')),
            _py(result.get('eps_trend')), json.dumps(result.get('eps_quarters')),
            _py(result.get('power_moves_count')), _py(result.get('alert_type')),
            _py(result.get('daily_volatility_pct')), _py(result.get('suggested_stop_pct')),
            _py(result.get('analysis_notes')),
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.error(f"Failed to store result for {result['symbol']}: {e}")


def run_scan(args):
    """Run the full market scan."""
    log.info("=" * 60)
    log.info("DOLPH SCAN_ONLINE — Market Scanner Starting")
    log.info("=" * 60)

    # Get stock universe
    if args['symbol']:
        stocks = [{'symbol': args['symbol'], 'name': '', 'sector': '', 'industry': '',
                    'exchange': 'US', 'currency': 'USD'}]
    else:
        stocks = get_full_scan_universe_v2() if args['full_market'] else get_scan_universe()

    # Enrich with IB scanner hot stocks (microcaps, unusual activity)
    if args['use_ib']:
        ib_stocks = scan_ib_hot_stocks(port=args['ib_port'])
        # Add IB stocks that aren't already in the universe
        existing = {s['symbol'] for s in stocks}
        new_from_ib = [s for s in ib_stocks if s['symbol'] not in existing]
        stocks.extend(new_from_ib)
        log.info(f'Added {len(new_from_ib)} new stocks from IB scanner (total: {len(stocks)})')

    if args['sector']:
        stocks = [s for s in stocks if args['sector'].lower() in s.get('sector', '').lower()]
        log.info(f"Filtered to sector '{args['sector']}': {len(stocks)} stocks")

    # Portfolio monitoring: compute trailing stops BEFORE main scan
    # (avoids yfinance rate limiting after scanning 7000+ stocks)
    portfolio_section = ''
    try:
        positions = []
        if args['use_ib']:
            positions = get_open_positions(port=args['ib_port'], client_id=96)
        if not positions:
            try:
                positions = get_open_positions(port=4001, client_id=96)
            except Exception:
                pass
        if positions:
            trailing = compute_trailing_stops(positions, analyze_stock)
            portfolio_section = format_portfolio_section([], trailing)
            log.info(f'Portfolio: {len(positions)} positions, {len(trailing)} trailing stops computed')
    except Exception as e:
        log.error(f'Portfolio monitoring (pre-scan) failed: {e}')

    # Scan each stock
    all_results = []
    opportunities = []
    total = len(stocks)

    for idx, stock in enumerate(stocks, 1):
        symbol = stock['symbol']
        if idx % 50 == 0 or idx == 1:
            log.info(f"Scanning {idx}/{total}: {symbol} ({stock.get('sector', '')})")

        result = analyze_stock(symbol)
        if result is None:
            continue

        # Enrich with universe metadata if yfinance didn't provide
        if not result.get('sector') and stock.get('sector'):
            result['sector'] = stock['sector']
        if not result.get('industry') and stock.get('industry'):
            result['industry'] = stock['industry']
        result['exchange'] = stock.get('exchange', 'US')

        # Store ALL results in DB
        if result.get('alert_type'):
            store_result(result)

        all_results.append(result)

        # Collect opportunities (2+ power moves or volume spike)
        if result['power_moves_count'] >= args['min_power_moves'] and result.get('alert_type'):
            # Deduplicate: skip if already seen with same or higher PM count
            existing = next((o for o in opportunities if o['symbol'] == result['symbol']), None)
            if existing:
                if result['power_moves_count'] > existing['power_moves_count']:
                    opportunities.remove(existing)
                    opportunities.append(result)
            else:
                opportunities.append(result)
            alert_type = result['alert_type']
            pm = result['power_moves_count']
            log.info(f"  ★ {alert_type}: {symbol} — {result.get('company_name', '')} "
                     f"(PM={pm}/3) {result.get('analysis_notes', '')}")

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info(f"SCAN COMPLETE: {len(all_results)}/{total} stocks analyzed")
    log.info(f"  Opportunities (PM>={args['min_power_moves']}): {len(opportunities)}")

    by_type = {}
    for o in opportunities:
        t = o.get('alert_type', 'UNKNOWN')
        by_type[t] = by_type.get(t, 0) + 1
    for t, c in sorted(by_type.items()):
        log.info(f"    {t}: {c}")
    log.info("=" * 60)

    # Send email alert
    # Cross-reference warnings with open positions (after scan)
    try:
        if positions and all_results:
            pos_alerts = check_positions_vs_warnings(positions, all_results)
            if pos_alerts:
                # Regenerate portfolio section with alerts
                trailing = [t for t in (trailing if 'trailing' in dir() else [])]
                portfolio_section = format_portfolio_section(pos_alerts, trailing)
                log.warning(f'POSITION ALERTS: {len(pos_alerts)} open positions have WARNING signals!')
                for a in pos_alerts:
                    log.warning(f'  {a["message"]}')
    except Exception as e:
        log.error(f'Portfolio warning cross-reference failed: {e}')

    # Screen for high-dividend undervalued stocks
    dividend_section = ''
    try:
        log.info('Screening for high-dividend undervalued stocks...')
        div_stocks = screen_high_dividend_undervalued(min_yield=7.0)
        if div_stocks:
            dividend_section = _format_dividend_section(div_stocks)
            log.info(f'Found {len(div_stocks)} high-dividend undervalued stocks')
    except Exception as e:
        log.error(f'Dividend screening failed: {e}')

    # Combine portfolio: dividend section + trailing stops
    if dividend_section and portfolio_section:
        portfolio_section = dividend_section + "\n" + portfolio_section
    elif dividend_section:
        portfolio_section = dividend_section

    if not args['no_email'] and (opportunities or portfolio_section):
        try:
            # Load email config directly from TradingPlatformSettings file
            _tps = {}
            tps_file = os.path.join(os.path.dirname(__file__), '..', '..', 'TradingPlatfomSettings-5.py')
            if not os.path.exists(tps_file):
                tps_file = os.path.join(os.path.dirname(__file__), '..', '..', 'TradingPlatfomSettings-0.py')
            with open(tps_file) as f:
                exec(f.read(), _tps)
            secrets = _tps.get('platform', {}).get('secrets', {})

            email_config = {
                'from_email': secrets['from_email'],
                'from_password': secrets['from_password'],
                'to_emails': secrets['to_emails'],
                'smtp_host': 'instaltic-com.correoseguro.dinaserver.com',
                'smtp_port': 587,
            }
            send_opportunity_alert(opportunities, email_config, portfolio_section=portfolio_section)
        except Exception as e:
            log.error(f"Email alert failed: {e}")
            # Print to stdout as fallback
            from ScanEngine.alert_email import _format_email_body
            print(_format_email_body(opportunities))

    # Print results table
    if opportunities:
        print()
        print(f"{'Symbol':8s} {'Name':25s} {'Sector':22s} {'Price':>8s} {'PM':>3s} {'Alert':15s} {'Notes'}")
        print("-" * 120)
        for o in sorted(opportunities, key=lambda x: -x['power_moves_count']):
            print(f"{o['symbol']:8s} {o.get('company_name', '')[:25]:25s} "
                  f"{o.get('sector', '')[:22]:22s} "
                  f"${o['close_price']:>7.2f} {o['power_moves_count']:>3d} "
                  f"{o.get('alert_type', ''):15s} "
                  f"{(o.get('analysis_notes') or '')[:50]}")

    return opportunities


if __name__ == '__main__':
    args = parse_args()
    run_scan(args)
