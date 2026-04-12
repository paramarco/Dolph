"""IB commission model — single source of truth for all fee calculations."""

_EU_EXCHANGES = frozenset({'IBIS', 'SBF', 'AEB', 'BM', 'BVME'})
_UK_EXCHANGES = frozenset({'LSE'})
_HK_EXCHANGES = frozenset({'SEHK'})
_JP_EXCHANGES = frozenset({'TSEJ'})


def ib_commission_per_side(quantity, primary_exchange):
    """IB commission per side based on the security's primary exchange.

    Returns the commission in the security's local currency.
    US:   max($1.00, qty * $0.005)  -- IB Fixed pricing
    EU:   3.00 EUR flat             -- IBIS, SBF, AEB, BM, BVME
    UK:   3.00 GBP flat             -- LSE
    HK:   18.00 HKD flat            -- SEHK
    JP:   80.00 JPY flat            -- TSEJ
    """
    pe = (primary_exchange or '').upper()
    if pe in _EU_EXCHANGES:
        return 3.0
    if pe in _UK_EXCHANGES:
        return 3.0
    if pe in _HK_EXCHANGES:
        return 18.0
    if pe in _JP_EXCHANGES:
        return 80.0
    # US default
    return max(1.0, quantity * 0.005)
