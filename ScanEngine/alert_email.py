#!/usr/bin/env python3
"""Email alert system for SCAN_ONLINE opportunities."""
import logging
import datetime as dt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

log = logging.getLogger("ScanEngine")


def send_opportunity_alert(opportunities, config, portfolio_section=None):
    """Send email with detected opportunities.

    Args:
        opportunities: list of dicts from power_moves.analyze_stock()
        config: dict with email settings (from_email, from_password, to_emails, smtp_host, smtp_port)
    """
    if not opportunities and not portfolio_section:
        return

    # Deduplicate before counting — only count BUY signals
    seen = {}
    for o in (opportunities or []):
        sym = o['symbol']
        if o.get('wyckoff_recommendation') != 'BUY':
            continue
        if sym not in seen or o['power_moves_count'] > seen[sym]['power_moves_count']:
            seen[sym] = o
    unique_count = len(seen)
    triples = sum(1 for o in seen.values() if o['power_moves_count'] >= 3)

    utc_now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M")
    if unique_count > 0:
        subject = f"🐬 Dolph detected {unique_count} Opportunities"
        if triples > 0:
            subject += f" ({triples} Triple Confirmation)"
    else:
        subject = f"🐬 Dolph Portfolio Update"
    subject += f" - {utc_now}"
    body = _format_email_body(opportunities) if opportunities else ""
    if portfolio_section:
        body += "\n\n" + portfolio_section

    try:
        from_email = config['from_email']
        from_password = config['from_password']
        to_emails = config['to_emails']
        smtp_host = config.get('smtp_host', 'instaltic-com.correoseguro.dinaserver.com')
        smtp_port = config.get('smtp_port', 587)

        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_emails, msg.as_string())
        server.quit()

        log.info(f"Alert email sent: {len(opportunities)} opportunities to {to_emails}")

    except Exception as e:
        log.error(f"Failed to send alert email: {e}")


def _append_stock_detail(lines, o):
    """Append formatted stock detail to lines (compact 3-line format)."""
    lines.append("")

    # Line 1: Symbol, name, sector, industry, power moves
    line1 = f"  {o['symbol']} — {o.get('company_name', '')} | "
    line1 += f"Sector: {o.get('sector', 'N/A')} | Industry: {o.get('industry', 'N/A')} | "
    line1 += f"Power Moves: {o['power_moves_count']}/3"
    lines.append(line1)

    # Line 2: Power move details (volume, breakout, EPS) with emojis
    pm_parts = []
    if o.get('volume_spike'):
        vol_emoji = '📊' if o.get('volume_direction') == 'GREEN' else '🔴'
        pm_parts.append(f"{vol_emoji} Volume Spike: {o['volume_today']:,} "
                       f"({o['volume_ratio']:.1f}x avg {o['volume_avg_50d']:,})")
    if o.get('breakout_detected'):
        pm_parts.append(f"📈 Heartbeat Breakout: above resistance ${o['resistance_level']:.2f} "
                       f"after {o['consolidation_months']:.0f} months "
                       f"(Range: ${o['support_level']:.2f} — ${o['resistance_level']:.2f}, "
                       f"{o['resistance_touches']} touches)")
    if o.get('eps_turned_positive') and o.get('eps_latest') is not None:
        eps_q = o.get('eps_quarters', [])
        eps_str = ", ".join(f"${q['eps']:.2f}" for q in eps_q[:8]) if eps_q else ""
        trend = o.get('eps_trend', '?')
        pm_parts.append(f"💰 EPS Inflection: turned positive (EPS): {eps_str} [{trend}]")
    elif o.get('eps_record') and o.get('eps_latest') is not None:
        eps_q = o.get('eps_quarters', [])
        eps_str = ", ".join(f"${q['eps']:.2f}" for q in eps_q[:8]) if eps_q else ""
        trend = o.get('eps_trend', '?')
        pm_parts.append(f"💰 Record Quarter (EPS): {eps_str} [{trend}]")
    if pm_parts:
        lines.append(f"  {' | '.join(pm_parts)}")

    # Line 3: Price, volatility, stop
    stop_pct = o.get('suggested_stop_pct', 15)
    stop_price = o['close_price'] * (1 - stop_pct / 100)
    line3 = f"  Price: ${o['close_price']:.2f} ({o['daily_change_pct']:+.2f}%) "
    line3 += f"Volatility: {o.get('daily_volatility_pct', 0):.2f}%/day → "
    line3 += f"Stop: {stop_pct}% below entry → ${stop_price:.2f}"
    lines.append(line3)

    # Line 4: Wyckoff phase + recommendation (padded for vertical alignment)
    phase = o.get('wyckoff_phase', '')
    rec = o.get('wyckoff_recommendation', '')
    reasoning = o.get('wyckoff_reasoning', '')
    if phase:
        phase_emoji = {'ACCUMULATION': '⬇️➡️', 'MARKUP': '⬆️', 'DISTRIBUTION': '⬆️➡️', 'MARKDOWN': '⬇️', 'SPRING': '⚡', 'UPTHRUST': '⚠️', 'TRANSITION': '⚪'}.get(phase, '⚪')
        rec_label = {'BUY': '✅ BUY ', 'SELL': '❌ SELL', 'HOLD': '⏸️ HOLD', 'AVOID': '❌ AVOID'}.get(rec, rec)
        lines.append(f"  Wyckoff: {phase_emoji} {phase:12s} | {rec_label:} — {reasoning}")


def _format_dividend_section(dividends):
    """Format high-dividend undervalued stocks section (BUY only, top 10)."""
    # Filter: only show Wyckoff BUY signals
    dividends = [d for d in dividends if d.get('wyckoff_recommendation') == 'BUY']
    dividends = dividends[:10]  # Top 10 by yield (already sorted)
    if not dividends:
        return ""
    lines = []
    lines.append(f"{'─' * 50}")
    lines.append(f"  👑 DIVIDEND KINGS ({len(dividends)} stocks, yield >7%)")
    lines.append(f"{'─' * 50}")
    lines.append("")
    lines.append(f"  {'Symbol':8s} {'Name':25s} {'Price':>8s} {'Yield':>6s} {'P/E':>6s} {'52wH':>8s} {'-%52w':>6s} {'Why undervalued'}")
    lines.append(f"  {'-' * 90}")
    for d in dividends:
        pe = f"{d['pe_ratio']:.1f}" if d['pe_ratio'] else "N/A"
        reasons = ", ".join(d.get('undervalued_reasons', []))
        lines.append(
            f"  {d['symbol']:8s} {d['company_name'][:25]:25s} "
            f"${d['current_price']:>7.2f} {d['dividend_yield']:>5.1f}% {pe:>6s} "
            f"${d['week52_high']:>7.2f} {d['below_52w_pct']:>5.1f}% "
            f"{reasons}"
        )
        # Wyckoff line
        phase = d.get('wyckoff_phase', '')
        rec = d.get('wyckoff_recommendation', '')
        reasoning = d.get('wyckoff_reasoning', '')
        if phase:
            phase_emoji = {'ACCUMULATION': '⬇️➡️', 'MARKUP': '⬆️', 'DISTRIBUTION': '⬆️➡️', 'MARKDOWN': '⬇️', 'SPRING': '⚡', 'UPTHRUST': '⚠️', 'TRANSITION': '⚪'}.get(phase, '⚪')
            rec_label = {'BUY': '✅ BUY ', 'SELL': '❌ SELL', 'HOLD': '⏸️ HOLD', 'AVOID': '❌ AVOID'}.get(rec, rec)
            lines.append(f"           Wyckoff: {phase_emoji} {phase:12s} | {rec_label} — {reasoning}")
    lines.append("")
    return "\n".join(lines)


def _format_email_body(opportunities):
    """Format opportunities into a readable email body."""
    # Deduplicate by symbol (keep highest power_moves_count)
    seen = {}
    for o in opportunities:
        sym = o['symbol']
        if sym not in seen or o['power_moves_count'] > seen[sym]['power_moves_count']:
            seen[sym] = o
    opportunities = sorted(seen.values(), key=lambda x: (-x['power_moves_count'], -x.get('volume_ratio', 0)))

    # Filter: only show BUY signals in opportunities
    buy_only = [o for o in opportunities if o.get('wyckoff_recommendation') == 'BUY']

    lines = []
    lines.append("🐬 Opportunities detected by Dolph:")
    lines.append("")

    # TRIPLE CONFIRMATION section first (3/3 Power Moves)
    triples = [o for o in buy_only if o['power_moves_count'] >= 3]
    if triples:
        lines.append(f"{'━' * 50}")
        lines.append(f"  ⭐ TRIPLE CONFIRMATION — {len(triples)} stocks (3/3 Power Moves)")
        lines.append(f"{'━' * 50}")
        for o in triples:
            _append_stock_detail(lines, o)
        lines.append("")

    # Group remaining by alert type
    for alert_type in ['OPPORTUNITY', 'SECOND_CHANCE']:
        group = [o for o in buy_only if o.get('alert_type') == alert_type and o['power_moves_count'] < 3]
        if not group:
            continue

        lines.append(f"{'─' * 50}")
        lines.append(f"  {alert_type} ({len(group)} stocks)")
        lines.append(f"{'─' * 50}")

        for o in group:
            _append_stock_detail(lines, o)

        lines.append("")



    return "\n".join(lines)
