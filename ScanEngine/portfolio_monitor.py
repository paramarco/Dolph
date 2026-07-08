#!/usr/bin/env python3
"""Portfolio monitor: cross-reference open IB positions with scan results.

- Alerts when a WARNING signal matches an open position (institutional selling!)
- Computes daily trailing stop suggestions for all open positions
- Includes in the scan email as a dedicated section
"""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
log = logging.getLogger("ScanEngine")

import psycopg2

_DB = dict(host='127.0.0.1', port=4713, user='dolph_user',
           password='dolph_password', dbname='dolph_db', sslmode='disable')


def _get_saved_stop(account, symbol):
    """Get the saved highest_price and stop_price from DB."""
    try:
        conn = psycopg2.connect(**_DB)
        cur = conn.cursor()
        cur.execute(
            "SELECT highest_price, stop_price, stop_pct FROM trailing_stops WHERE account = %s AND symbol = %s",
            (account, symbol))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {'highest_price': float(row[0]), 'stop_price': float(row[1]), 'stop_pct': float(row[2])}
    except Exception as e:
        log.debug(f"Failed to get saved stop for {symbol}: {e}")
    return None


def _save_stop(account, symbol, side, entry_price, highest_price, stop_price, stop_pct):
    """Save or update the trailing stop high watermark in DB."""
    try:
        conn = psycopg2.connect(**_DB)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trailing_stops (account, symbol, side, entry_price, highest_price, stop_price, stop_pct, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (account, symbol)
            DO UPDATE SET highest_price = EXCLUDED.highest_price,
                         stop_price = EXCLUDED.stop_price,
                         stop_pct = EXCLUDED.stop_pct,
                         entry_price = EXCLUDED.entry_price,
                         side = EXCLUDED.side,
                         updated_at = NOW()
        """, (account, symbol, side, entry_price, highest_price, stop_price, stop_pct))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.debug(f"Failed to save stop for {symbol}: {e}")


def get_open_positions(host='127.0.0.1', port=4001, client_id=96):
    """Fetch all open positions from IB."""
    try:
        sys.path.insert(0, '/opt/venv/lib/python3.11/site-packages')
        from ib_insync import IB
    except ImportError:
        log.error("ib_insync not available")
        return []

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as e:
        log.warning(f"IB connection failed for portfolio monitor: {e}")
        return []

    positions = []
    for p in ib.positions():
        if p.position == 0:
            continue
        positions.append({
            'account': p.account,
            'symbol': p.contract.symbol,
            'quantity': float(p.position),
            'avg_cost': float(p.avgCost),
            'side': 'LONG' if p.position > 0 else 'SHORT',
            'currency': p.contract.currency,
            'exchange': p.contract.primaryExchange or '',
        })

    ib.disconnect()
    log.info(f"Portfolio monitor: {len(positions)} open positions")
    return positions


def check_positions_vs_warnings(positions, scan_results):
    """Cross-reference open positions with WARNING scan results.

    Returns list of alerts for positions that have WARNING signals
    (institutional selling detected on a stock you hold).
    """
    # Get all WARNING symbols from scan
    warning_symbols = {}
    for r in scan_results:
        if r.get('alert_type') == 'WARNING' and r.get('volume_spike'):
            warning_symbols[r['symbol']] = r

    alerts = []
    for pos in positions:
        sym = pos['symbol']
        # Check exact match and yfinance variants (.L, .DE, etc.)
        warning = warning_symbols.get(sym)
        if not warning:
            # Try with common suffixes
            for suffix in ['', '.L', '.DE', '.PA', '.MC', '.MI', '.AS', '.SW', '.ST']:
                warning = warning_symbols.get(f"{sym}{suffix}")
                if warning:
                    break

        if warning and pos['side'] == 'LONG':
            alerts.append({
                'position': pos,
                'warning': warning,
                'message': (f"DANGER: {sym} has massive selling volume "
                           f"({warning['volume_ratio']:.1f}x avg) while you hold "
                           f"{abs(pos['quantity']):.0f} shares LONG at ${pos['avg_cost']:.2f}"),
            })

    return alerts


def compute_trailing_stops(positions, analyze_stock_fn):
    """Compute trailing stop suggestions for all open positions.

    Uses current price and volatility to suggest stop levels.
    """
    import warnings as _w
    _w.filterwarnings('ignore')

    results = []
    for pos in positions:
        sym = pos['symbol']
        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            hist = ticker.history(period='1y', interval='1d')
            if hist is None or len(hist) < 5:
                continue

            current_price = float(hist['Close'].iloc[-1])
            entry_price = pos['avg_cost']
            qty = abs(pos['quantity'])

            # Compute volatility
            returns = hist['Close'].pct_change().dropna().tail(20)
            daily_vol = float(returns.std() * 100)

            # Determine stop percentage based on volatility
            if daily_vol >= 3.0:
                stop_pct = 22
            elif daily_vol >= 1.5:
                stop_pct = 12
            else:
                stop_pct = 7

            # Trailing stop with high watermark: stop NEVER moves against you
            if pos['side'] == 'LONG':
                new_stop = current_price * (1 - stop_pct / 100)
                pnl_pct = (current_price / entry_price - 1) * 100
                pnl_abs = (current_price - entry_price) * qty
            else:
                new_stop = current_price * (1 + stop_pct / 100)
                pnl_pct = (1 - current_price / entry_price) * 100
                pnl_abs = (entry_price - current_price) * qty

            # High watermark: stop only moves in your favor, never back
            saved = _get_saved_stop(pos['account'], sym)
            highest_price = current_price
            stop_price = new_stop

            if saved:
                if pos['side'] == 'LONG':
                    highest_price = max(current_price, saved['highest_price'])
                    best_stop = highest_price * (1 - stop_pct / 100)
                    stop_price = max(best_stop, saved['stop_price'])
                else:
                    highest_price = min(current_price, saved['highest_price'])
                    best_stop = highest_price * (1 + stop_pct / 100)
                    stop_price = min(best_stop, saved['stop_price'])

            _save_stop(pos['account'], sym, pos['side'], entry_price,
                      highest_price, stop_price, stop_pct)

            # Wyckoff phase
            wyckoff_info = {'phase': '', 'recommendation': '', 'reasoning': ''}
            try:
                from ScanEngine.power_moves import detect_wyckoff_phase
                wyckoff_info = detect_wyckoff_phase(hist, has_position=True)
            except Exception:
                pass

            results.append({
                'account': pos['account'],
                'symbol': sym,
                'side': pos['side'],
                'quantity': qty,
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl_pct': round(pnl_pct, 2),
                'pnl_abs': round(pnl_abs, 2),
                'currency': pos['currency'],
                'daily_volatility': round(daily_vol, 2),
                'stop_pct': stop_pct,
                'stop_price': round(stop_price, 2),
                'wyckoff_phase': wyckoff_info.get('phase', ''),
                'wyckoff_recommendation': wyckoff_info.get('recommendation', ''),
                'wyckoff_reasoning': wyckoff_info.get('reasoning', ''),
            })
        except Exception as e:
            log.debug(f"Trailing stop calc failed for {sym}: {e}")

    return results


def format_portfolio_section(alerts, trailing_stops):
    """Format portfolio monitoring section for the email."""
    lines = []

    if alerts:
        lines.append(f"{'=' * 50}")
        lines.append(f"  !! POSITION ALERTS ({len(alerts)} warnings) !!")
        lines.append(f"{'=' * 50}")
        for a in alerts:
            lines.append(f"")
            lines.append(f"  {a['message']}")
            w = a['warning']
            lines.append(f"  Volume: {w.get('volume_today', 0):,} ({w['volume_ratio']:.1f}x avg)")
            lines.append(f"  ACTION: Consider tightening stop loss or reducing position")
        lines.append("")

    if trailing_stops:
        lines.append(f"{'─' * 50}")
        lines.append(f"  Trailing Stop Suggestions ({len(trailing_stops)} positions)")
        lines.append(f"{'─' * 50}")
        lines.append(f"")
        lines.append(f"  {'Symbol':8s} {'Side':5s} {'Entry':>8s} {'Now':>8s} {'P&L%':>7s} {'P&L$':>9s} {'Vol%':>5s} {'Stop':>8s}")
        lines.append(f"  {'-' * 68}")

        # Sort: alerts first, then by P&L
        for s in sorted(trailing_stops, key=lambda x: x['pnl_pct']):
            pnl_marker = '🔴' if s['pnl_pct'] < -10 else ''
            lines.append(
                f"  {s['symbol']:8s} {s['side']:5s} "
                f"${s['entry_price']:>7.2f} ${s['current_price']:>7.2f} "
                f"{s['pnl_pct']:>+6.1f}% ${s['pnl_abs']:>+8.2f} "
                f"{s['daily_volatility']:>4.1f}% ${s['stop_price']:>7.2f} {pnl_marker}"
            )
            # Wyckoff line for each position (padded for vertical alignment)
            phase = s.get('wyckoff_phase', '')
            rec = s.get('wyckoff_recommendation', '')
            reasoning = s.get('wyckoff_reasoning', '')
            if phase:
                rec_label = {'BUY': '✅ BUY ', 'SELL': '❌ SELL', 'HOLD': '⏸️ HOLD', 'AVOID': '❌ AVOID'}.get(rec, rec)
                phase_emoji = {'ACCUMULATION': '⬇️➡️', 'MARKUP': '⬆️', 'DISTRIBUTION': '⬆️➡️', 'MARKDOWN': '⬇️', 'SPRING': '⚡', 'UPTHRUST': '⚠️', 'TRANSITION': '⚪'}.get(phase, '⚪')
                lines.append(f"           Wyckoff: {phase_emoji} {phase:12s} | {rec_label} — {reasoning}")
        lines.append("")

    return "\n".join(lines)
