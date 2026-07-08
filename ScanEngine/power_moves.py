#!/usr/bin/env python3
"""Three Power Moves detection for the SCAN_ONLINE market scanner.

Power Move 1: Institutional Buying (Volume Spike 300%+)
Power Move 2: Heartbeat Pattern (Consolidation Breakout)
Power Move 3: Record Quarter (EPS inflection)
"""
import logging
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')
log = logging.getLogger("ScanEngine")

# ---------- Configuration ----------
VOLUME_SPIKE_THRESHOLD = 3.0     # 300% of 50-day average
VOLUME_AVG_DAYS = 50             # Days for average volume calculation
MIN_CONSOLIDATION_MONTHS = 3     # Minimum sideways duration
MIN_RESISTANCE_TOUCHES = 3       # Minimum touches on resistance
RESISTANCE_TOLERANCE = 0.02      # 2% tolerance for "touching" resistance
EPS_LOOKBACK_QUARTERS = 8        # Quarters to analyze for EPS trend


def analyze_stock(symbol):
    """Run all Three Power Moves analysis on a single stock.

    Returns dict with full analysis or None on error.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Fetch 2 years of daily data for heartbeat pattern
        hist = ticker.history(period='2y', interval='1d')
        if hist is None or len(hist) < 60:
            return None

        info = ticker.info or {}
        # Use current/live price if available (market open), else last daily close
        last_daily_close = float(hist['Close'].iloc[-1])
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if current_price and float(current_price) > 0:
            close_price = float(current_price)
        else:
            close_price = last_daily_close

        # Include today's intraday volume if market is open
        intraday_vol = 0
        try:
            hist_1d = ticker.history(period='1d', interval='1d')
            if hist_1d is not None and len(hist_1d) > 0:
                last_bar_date = hist_1d.index[-1].date()
                import datetime
                if last_bar_date >= datetime.date.today():
                    intraday_vol = int(hist_1d['Volume'].iloc[-1])
        except Exception:
            pass

        result = {
            'symbol': symbol,
            'company_name': info.get('shortName', info.get('longName', '')),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'currency': info.get('currency', 'USD'),
            'close_price': close_price,
            'daily_change_pct': round((close_price / last_daily_close - 1) * 100, 4) if last_daily_close > 0 else 0,
            '_intraday_vol': intraday_vol,
            '_last_daily_close': last_daily_close,
        }

        # Power Move 1: Volume Spike
        vol_result = _detect_volume_spike(hist)
        result.update(vol_result)

        # Power Move 2: Heartbeat Pattern
        hb_result = _detect_heartbeat_breakout(hist)
        result.update(hb_result)

        # Power Move 3: Record Quarter EPS
        eps_result = _detect_record_quarter(ticker)
        result.update(eps_result)

        # Wyckoff phase analysis
        wyckoff = detect_wyckoff_phase(hist)
        result['wyckoff_phase'] = wyckoff['phase']
        result['wyckoff_recommendation'] = wyckoff['recommendation']
        result['wyckoff_reasoning'] = wyckoff['reasoning']

        # Overall assessment
        pm_count = sum([
            result['volume_spike'],
            result['breakout_detected'],
            result['eps_record'] or result['eps_turned_positive'],
        ])
        result['power_moves_count'] = pm_count

        # Alert type
        result['alert_type'] = _classify_alert(result)

        # Trailing stop suggestion
        result.update(_suggest_stop(hist))

        # Analysis notes
        result['analysis_notes'] = _build_notes(result)

        return result

    except Exception as e:
        log.debug(f"{symbol}: analysis failed: {e}")
        return None


def _daily_change(hist):
    """Percentage change of last close vs previous close."""
    if len(hist) < 2:
        return 0.0
    return round((hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100, 4)


# =====================================================
# POWER MOVE 1: Volume Spike (Institutional Buying)
# =====================================================

def _detect_volume_spike(hist):
    """Detect if last trading day had extreme volume (300%+ of 50-day avg)."""
    result = {
        'volume_today': 0,
        'volume_avg_50d': 0,
        'volume_ratio': 0.0,
        'volume_direction': 'NEUTRAL',
        'volume_spike': False,
    }

    if len(hist) < VOLUME_AVG_DAYS + 1:
        return result

    vol = hist['Volume'].values
    closes = hist['Close'].values
    opens = hist['Open'].values

    vol_today = int(vol[-1])
    # If intraday volume is available and higher, use it (market is open)
    intraday = hist.attrs.get('_intraday_vol', 0) if hasattr(hist, 'attrs') else 0
    if intraday > vol_today:
        vol_today = intraday
    vol_avg = int(np.mean(vol[-VOLUME_AVG_DAYS - 1:-1]))  # Exclude today
    ratio = vol_today / vol_avg if vol_avg > 0 else 0

    # Direction: green (up) or red (down) day
    is_up = closes[-1] > opens[-1]
    direction = 'GREEN' if is_up else 'RED'

    result['volume_today'] = vol_today
    result['volume_avg_50d'] = vol_avg
    result['volume_ratio'] = round(ratio, 2)
    result['volume_direction'] = direction
    result['volume_spike'] = ratio >= VOLUME_SPIKE_THRESHOLD

    return result


# =====================================================
# POWER MOVE 2: Heartbeat Pattern (Consolidation Breakout)
# =====================================================

def _detect_heartbeat_breakout(hist):
    """Detect consolidation range and breakout above resistance."""
    result = {
        'resistance_level': 0.0,
        'support_level': 0.0,
        'consolidation_months': 0.0,
        'resistance_touches': 0,
        'breakout_detected': False,
    }

    if len(hist) < 60:  # Need at least 3 months
        return result

    closes = hist['Close'].values
    highs = hist['High'].values
    lows = hist['Low'].values
    n = len(closes)

    # Find consolidation range using the last 6-24 months
    # Look for the longest period where price stayed within a channel
    best_range = None
    best_duration = 0

    # Try different lookback windows (3 months to 2 years)
    for lookback_days in [60, 90, 130, 180, 260, 390, 500]:
        if lookback_days >= n:
            continue

        window = closes[-lookback_days:]
        w_highs = highs[-lookback_days:]
        w_lows = lows[-lookback_days:]

        # Resistance = rolling max (excluding last 5 days for breakout detection)
        resistance = np.percentile(w_highs[:-5], 95)
        support = np.percentile(w_lows[:-5], 5)

        channel_width = (resistance - support) / resistance if resistance > 0 else 0

        # Valid consolidation: channel width < 30% (not too wide)
        if 0.03 < channel_width < 0.30:
            # Count resistance touches (within 2% of resistance)
            touches = sum(1 for h in w_highs[:-5]
                          if abs(h - resistance) / resistance < RESISTANCE_TOLERANCE)

            duration_months = lookback_days / 21  # ~21 trading days/month

            if touches >= MIN_RESISTANCE_TOUCHES and duration_months >= MIN_CONSOLIDATION_MONTHS:
                if duration_months > best_duration:
                    best_duration = duration_months
                    best_range = {
                        'resistance': resistance,
                        'support': support,
                        'touches': touches,
                        'duration_months': duration_months,
                    }

    if best_range is None:
        return result

    result['resistance_level'] = round(best_range['resistance'], 4)
    result['support_level'] = round(best_range['support'], 4)
    result['consolidation_months'] = round(best_range['duration_months'], 1)
    result['resistance_touches'] = best_range['touches']

    # Check breakout: last close above resistance
    last_close = closes[-1]
    breakout = last_close > best_range['resistance']

    # Volume confirmation on breakout day
    if breakout and len(hist) > VOLUME_AVG_DAYS:
        vol_ratio = hist['Volume'].iloc[-1] / hist['Volume'].iloc[-VOLUME_AVG_DAYS - 1:-1].mean()
        # Breakout requires some volume confirmation (at least 1.5x avg)
        if vol_ratio < 1.5:
            breakout = False

    result['breakout_detected'] = breakout
    return result


# =====================================================
# POWER MOVE 3: Record Quarter EPS
# =====================================================

def _detect_record_quarter(ticker):
    """Analyze quarterly EPS for record quarter or positive inflection."""
    result = {
        'eps_latest': None,
        'eps_record': False,
        'eps_turned_positive': False,
        'eps_trend': 'UNKNOWN',
        'eps_quarters': None,
    }

    try:
        q_income = ticker.quarterly_income_stmt
        if q_income is None or q_income.empty:
            return result

        # Get Diluted EPS (preferred) or Basic EPS
        eps_key = None
        for key in ['Diluted EPS', 'Basic EPS']:
            if key in q_income.index:
                eps_key = key
                break
        if eps_key is None:
            return result

        eps_series = q_income.loc[eps_key].dropna()
        if len(eps_series) < 2:
            return result

        # Take last 8 quarters, most recent first
        eps_values = eps_series.head(EPS_LOOKBACK_QUARTERS).values.tolist()
        eps_dates = [str(d.date()) if hasattr(d, 'date') else str(d)
                     for d in eps_series.head(EPS_LOOKBACK_QUARTERS).index]

        result['eps_quarters'] = [
            {'date': d, 'eps': round(float(e), 4)}
            for d, e in zip(eps_dates, eps_values)
        ]

        latest = float(eps_values[0])
        result['eps_latest'] = round(latest, 4)

        # Check record quarter: is latest EPS the highest ever?
        if len(eps_values) >= 4:
            historical_max = max(eps_values[1:])  # Exclude latest
            result['eps_record'] = latest > historical_max and latest > 0

        # Check positive inflection: transition from negative to positive
        if len(eps_values) >= 2:
            previous = float(eps_values[1])
            result['eps_turned_positive'] = previous < 0 and latest > 0

        # Classify trend
        if result['eps_turned_positive']:
            result['eps_trend'] = 'VERY_STRONG'
        elif result['eps_record']:
            result['eps_trend'] = 'STRONG'
        elif len(eps_values) >= 3 and all(eps_values[i] >= eps_values[i + 1]
                                           for i in range(min(3, len(eps_values) - 1))):
            result['eps_trend'] = 'MODERATE'  # Increasing but not record
        else:
            result['eps_trend'] = 'WEAK'

    except Exception as e:
        log.debug(f"EPS analysis failed: {e}")

    return result


# =====================================================
# ALERT CLASSIFICATION
# =====================================================

def _classify_alert(result):
    """Classify the alert type based on Power Moves detected."""
    pm = result['power_moves_count']
    vol_spike = result['volume_spike']
    vol_dir = result['volume_direction']
    breakout = result['breakout_detected']

    # WARNING: massive selling volume on down day
    if vol_spike and vol_dir == 'RED':
        return 'WARNING'

    # OPPORTUNITY: 2+ power moves with volume spike required
    if pm >= 2 and vol_spike and vol_dir == 'GREEN':
        return 'OPPORTUNITY'

    # SECOND_CHANCE: breakout detected but price pulled back near resistance
    if breakout and not vol_spike:
        return 'SECOND_CHANCE'

    # Single power move → MONITOR
    if pm >= 1:
        return 'MONITOR'

    return None


def _suggest_stop(hist):
    """Suggest trailing stop percentage based on stock volatility."""
    if len(hist) < 20:
        return {'daily_volatility_pct': 0, 'suggested_stop_pct': 15}

    # Daily returns volatility (last 20 days)
    returns = hist['Close'].pct_change().dropna().tail(20)
    daily_vol = returns.std() * 100  # As percentage

    # Map volatility to stop level
    if daily_vol >= 3.0:
        stop_pct = 22  # High vol: wider stop
    elif daily_vol >= 1.5:
        stop_pct = 12  # Medium vol
    else:
        stop_pct = 7   # Low vol: tighter stop

    return {
        'daily_volatility_pct': round(daily_vol, 4),
        'suggested_stop_pct': stop_pct,
    }


def _build_notes(result):
    """Build human-readable analysis notes."""
    notes = []

    if result['volume_spike']:
        dir_emoji = '📊' if result['volume_direction'] == 'GREEN' else '🔴'
        notes.append(f"Volume Spike: {result['volume_ratio']:.1f}x avg "
                     f"({result['volume_direction']} day)")

    if result['breakout_detected']:
        notes.append(f"📈 Heartbeat Breakout: above resistance ${result['resistance_level']:.2f} "
                     f"after {result['consolidation_months']:.0f} months consolidation "
                     f"({result['resistance_touches']} touches)")

    eps_trend = result.get('eps_trend', 'UNKNOWN')
    if eps_trend in ('STRONG', 'VERY_STRONG'):
        eps = result.get('eps_latest', 0)
        if result['eps_turned_positive']:
            notes.append(f"💰 EPS Inflection: turned positive (${eps:.2f})")
        else:
            notes.append(f"💰 Record Quarter EPS: ${eps:.2f}")

    # Add stop suggestion
    stop_pct = result.get('suggested_stop_pct', 15)
    stop_price = result.get('close_price', 0) * (1 - stop_pct / 100)
    if stop_price > 0:
        notes.append(f"Stop: {stop_pct}% → ${stop_price:.2f}")

    if not notes:
        return None

    return ' | '.join(notes)


# =====================================================
# WYCKOFF PHASE DETECTION
# =====================================================

def detect_wyckoff_phase(hist, has_position=False):
    """Detect Wyckoff market phase from daily OHLCV data.
    
    Phases:
      - ACCUMULATION: sideways range after decline, smart money buying
      - MARKUP (Trend Up): price trending upward after accumulation
      - DISTRIBUTION: sideways range after advance, smart money selling
      - MARKDOWN (Trend Down): price trending downward after distribution
      - SPRING (False breakdown): price briefly breaks below support then recovers
      - UPTHRUST (False breakout): price briefly breaks above resistance then falls
    
    Returns dict with phase, recommendation, and reasoning.
    """
    if hist is None or len(hist) < 60:
        return {'phase': 'UNKNOWN', 'recommendation': 'HOLD', 'reasoning': 'Insufficient data'}

    closes = hist['Close'].values
    volumes = hist['Volume'].values
    highs = hist['High'].values
    lows = hist['Low'].values
    n = len(closes)

    # Current price and moving averages
    current = closes[-1]
    ma50 = np.mean(closes[-50:]) if n >= 50 else np.mean(closes)
    ma200 = np.mean(closes[-200:]) if n >= 200 else np.mean(closes[-min(n, 100):])

    # Price trend (last 3 months vs last 6 months)
    price_3m = closes[-min(63, n):]
    price_6m = closes[-min(126, n):]
    trend_3m = (price_3m[-1] / price_3m[0] - 1) * 100 if len(price_3m) > 1 else 0
    trend_6m = (price_6m[-1] / price_6m[0] - 1) * 100 if len(price_6m) > 1 else 0

    # Volatility / range analysis (last 3 months)
    recent_high = np.max(highs[-63:]) if n >= 63 else np.max(highs)
    recent_low = np.min(lows[-63:]) if n >= 63 else np.min(lows)
    channel_width = (recent_high - recent_low) / current * 100

    # Volume trend
    vol_recent = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
    vol_prior = np.mean(volumes[-50:-20]) if n >= 50 else np.mean(volumes[:max(1, n - 20)])
    vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0

    # Price position within recent range
    range_position = (current - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

    # MA slope (trend direction)
    if n >= 55:
        ma50_slope = (np.mean(closes[-5:]) - np.mean(closes[-55:-50])) / np.mean(closes[-55:-50]) * 100
    else:
        ma50_slope = trend_3m

    # Spring detection: price recently dipped below support then recovered
    support_3m = np.percentile(lows[-63:], 5) if n >= 63 else np.min(lows)
    recent_min = np.min(lows[-10:]) if n >= 10 else lows[-1]
    spring = recent_min < support_3m and current > support_3m * 1.02

    # Upthrust detection: price recently broke above resistance then fell
    resistance_3m = np.percentile(highs[-63:], 95) if n >= 63 else np.max(highs)
    recent_max = np.max(highs[-10:]) if n >= 10 else highs[-1]
    upthrust = recent_max > resistance_3m and current < resistance_3m * 0.98

    # Determine phase
    phase = 'UNKNOWN'
    recommendation = 'HOLD'
    reasoning = ''

    if spring:
        phase = 'SPRING'
        recommendation = 'BUY'
        reasoning = f'False breakdown below ${support_3m:.2f} then recovered to ${current:.2f} — classic Wyckoff spring, strong buy signal'

    elif upthrust:
        phase = 'UPTHRUST'
        recommendation = 'SELL' if has_position else 'AVOID'
        reasoning = f'False breakout above ${resistance_3m:.2f} then fell back to ${current:.2f} — Wyckoff upthrust, distribution likely'

    elif channel_width < 15 and abs(trend_3m) < 8:
        # Sideways range
        if trend_6m < -10:
            phase = 'ACCUMULATION'
            if vol_trend > 1.2 and range_position > 0.6:
                recommendation = 'BUY'
                reasoning = f'Sideways after {trend_6m:.0f}% decline, volume rising {vol_trend:.1f}x, price near top of range — accumulation with demand'
            else:
                recommendation = 'HOLD'
                reasoning = f'Sideways range after {trend_6m:.0f}% decline, channel {channel_width:.0f}% wide — accumulation phase, wait for breakout'
        elif trend_6m > 10:
            phase = 'DISTRIBUTION'
            if vol_trend > 1.2 and range_position < 0.4:
                recommendation = 'SELL' if has_position else 'AVOID'
                reasoning = f'Sideways after {trend_6m:.0f}% advance, volume rising {vol_trend:.1f}x, price near bottom of range — distribution with supply'
            else:
                recommendation = 'HOLD' if has_position else 'AVOID'
                reasoning = f'Sideways range after {trend_6m:.0f}% advance, channel {channel_width:.0f}% wide — distribution phase, watch for breakdown'
        else:
            phase = 'ACCUMULATION' if current < ma200 else 'DISTRIBUTION'
            recommendation = 'HOLD'
            reasoning = f'Sideways range ({channel_width:.0f}% wide), unclear trend — monitor for breakout direction'

    elif ma50_slope > 2 and current > ma50:
        phase = 'MARKUP'
        if trend_3m > 15:
            recommendation = 'HOLD' if has_position else 'BUY'
            reasoning = f'Strong uptrend +{trend_3m:.0f}% (3m), above MA50 — markup phase, {"trail stops higher" if has_position else "enter with trailing stop"}'
        else:
            recommendation = 'BUY'
            reasoning = f'Uptrend +{trend_3m:.0f}% (3m), price above MA50, MA slope positive — early markup, good entry'

    elif ma50_slope < -2 and current < ma50:
        phase = 'MARKDOWN'
        if trend_3m < -15:
            recommendation = 'SELL' if has_position else 'AVOID'
            reasoning = f'Strong downtrend {trend_3m:.0f}% (3m), below MA50 — markdown phase, {"exit or reduce" if has_position else "do not enter"}'
        else:
            recommendation = 'HOLD' if has_position else 'AVOID'
            reasoning = f'Downtrend {trend_3m:.0f}% (3m), below MA50 — markdown phase, wait for accumulation signs'

    else:
        # Transitional
        if current > ma50 and current > ma200:
            phase = 'MARKUP'
            recommendation = 'HOLD'
            reasoning = f'Above MA50 and MA200, trend +{trend_3m:.0f}% (3m) — markup continuation'
        elif current < ma50 and current < ma200:
            phase = 'MARKDOWN'
            recommendation = 'HOLD'
            reasoning = f'Below MA50 and MA200, trend {trend_3m:.0f}% (3m) — markdown, caution'
        else:
            phase = 'TRANSITION'
            recommendation = 'HOLD'
            reasoning = f'Between MA50 and MA200, trend {trend_3m:.0f}% (3m) — transitional phase'

    return {
        'phase': phase,
        'recommendation': recommendation,
        'reasoning': reasoning,
    }


def screen_high_dividend_undervalued(symbols=None, min_yield=7.0):
    """Screen for stocks with high dividend yield (>7%) that are undervalued.
    
    Undervalued criteria:
      - P/E ratio below sector average or < 15
      - Price below 52-week high by > 20%
      - Price below analyst target price
    
    Returns list of dicts with dividend and valuation data.
    """
    import yfinance as yf
    
    if symbols is None:
        # Default: scan S&P 500 + STOXX 600 for high dividend
        try:
            from ScanEngine.stock_universe import fetch_sp500, fetch_stoxx600
            sp500 = fetch_sp500()
            stoxx = fetch_stoxx600()
            symbols = [s['symbol'] for s in sp500 + stoxx]
        except Exception:
            symbols = []
    
    results = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            if current_price <= 0:
                continue

            # Calculate REAL annual yield from dividend history
            # Uses exactly 365 days, filters out extraordinary dividends (>3x median)
            div_yield_pct = 0
            try:
                divs = t.dividends
                if divs is not None and len(divs) > 0:
                    from datetime import datetime, timedelta
                    one_year_ago = datetime.now() - timedelta(days=365)
                    recent_divs = divs[divs.index >= str(one_year_ago.date())]
                    if len(recent_divs) > 0:
                        # Filter out extraordinary dividends
                        if len(recent_divs) >= 2:
                            # Multiple payments: filter any >3x the median
                            median_div = float(recent_divs.median())
                            if median_div > 0:
                                ordinary_divs = recent_divs[recent_divs <= median_div * 3]
                            else:
                                ordinary_divs = recent_divs
                        else:
                            # Single payment: skip if >20% of stock price (likely special/liquidating)
                            single_div = float(recent_divs.iloc[0])
                            if single_div > current_price * 0.20:
                                ordinary_divs = recent_divs.iloc[0:0]  # empty
                            else:
                                ordinary_divs = recent_divs
                        annual_div = float(ordinary_divs.sum())
                        if annual_div > 0:
                            calc_yield = annual_div / current_price * 100
                            # Sanity cap: yields above 25% are almost certainly data errors
                            if calc_yield <= 25:
                                div_yield_pct = calc_yield
            except Exception:
                pass
            
            # Fallback to yfinance reported yield if history not available
            # WARNING: yfinance dividendYield can be wildly wrong (e.g. GOOG 0.24 = "24%")
            # Only use trailingAnnualDividendYield (which is annual_div/price, already a ratio)
            if div_yield_pct == 0:
                dy = info.get('trailingAnnualDividendYield') or 0
                fallback_yield = dy * 100 if dy < 1 else dy
                if fallback_yield <= 25:
                    div_yield_pct = fallback_yield
            
            if div_yield_pct < min_yield:
                continue
            
            pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0
            week52_high = info.get('fiftyTwoWeekHigh') or 0
            target_price = info.get('targetMeanPrice') or 0
            
            # Undervalued checks
            below_52w = ((week52_high - current_price) / week52_high * 100) if week52_high > 0 else 0
            below_target = ((target_price - current_price) / current_price * 100) if target_price > 0 else 0
            pe_undervalued = pe_ratio > 0 and pe_ratio < 15
            
            is_undervalued = (below_52w > 20) or (below_target > 15) or pe_undervalued
            
            if not is_undervalued:
                continue
            
            # Wyckoff analysis for dividend stock
            wyckoff = {'phase': '', 'recommendation': '', 'reasoning': ''}
            try:
                hist = t.history(period='1y')
                if hist is not None and len(hist) >= 50:
                    wyckoff = detect_wyckoff_phase(hist, has_position=False)
            except Exception:
                pass

            results.append({
                'symbol': sym,
                'company_name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'current_price': round(current_price, 2),
                'dividend_yield': round(div_yield_pct, 2),
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                'week52_high': round(week52_high, 2),
                'below_52w_pct': round(below_52w, 1),
                'target_price': round(target_price, 2) if target_price else None,
                'below_target_pct': round(below_target, 1) if target_price else None,
                'undervalued_reasons': [],
                'wyckoff_phase': wyckoff['phase'],
                'wyckoff_recommendation': wyckoff['recommendation'],
                'wyckoff_reasoning': wyckoff['reasoning'],
            })
            
            r = results[-1]
            if below_52w > 20:
                r['undervalued_reasons'].append(f'{below_52w:.0f}% below 52w high')
            if below_target > 15:
                r['undervalued_reasons'].append(f'{below_target:.0f}% below target ${target_price:.0f}')
            if pe_undervalued:
                r['undervalued_reasons'].append(f'Low P/E {pe_ratio:.1f}')
                
        except Exception:
            continue
    
    results.sort(key=lambda x: -x['dividend_yield'])
    return results
