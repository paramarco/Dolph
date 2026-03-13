from ib_insync import Trade
from datetime import datetime, timezone

class OrderIB:

    def __init__(self, trade: Trade):
        """
        Custom wrapper for Interactive Brokers Trade object.
        """
        self.id = trade.order.orderId
        self.symbol = trade.contract.symbol
        self.status = trade.orderStatus.status
        self.type = trade.order.orderType
        self.side = 'buy' if trade.order.action.lower() == 'buy' else 'sell'
        self._raw = trade  # Keep the original Trade object for any advanced usage
        self.order = trade.order
        self.account = trade.order.account    # cuenta IB que posee esta orden
        # Store the creation time as UTC timestamp
        self.time = datetime.now(timezone.utc)

    def __eq__(self, other):
        """
        Compare orders based on their 'id' attribute.
        """
        if isinstance(other, OrderIB):
            return self.id == other.id
        return False

    # ib_insync sentinel for "not set"
    _UNSET_DOUBLE = 1.7976931348623157e+308

    def __str__(self):
        """
        String representation for debugging.
        """
        # Extraer el precio según el tipo de orden
        price = None
        try:
            order = self._raw.order
            if order.orderType in ('STP', 'STP LMT'):
                aux = getattr(order, 'auxPrice', None)
                if isinstance(aux, (int, float)) and 0 < aux < self._UNSET_DOUBLE:
                    price = aux
            if price is None:
                lmt = getattr(order, 'lmtPrice', None)
                if isinstance(lmt, (int, float)) and 0 < lmt < self._UNSET_DOUBLE:
                    price = lmt
            if price is None:
                avg = getattr(self._raw.orderStatus, 'avgFillPrice', None)
                if isinstance(avg, (int, float)) and avg > 0:
                    price = avg
        except Exception:
            pass
        
        msg = f"OrderIB(id={self.id}, symbol={self.symbol}, side={self.side}, "
        msg += f"type={self.type}, "
        if price is not None:
            msg += f"price={price}, "
        msg += f"status={self.status}, UTC-time={self.time.isoformat()})"
        return msg

    def __getattr__(self, name):
        """
        Custom attribute handler for additional logic.
        """
        if name == 'seccode':
            return self.symbol  # Map 'seccode' to 'symbol'
        if name == 'buysell':
            return self.side  # Map 'buysell' to 'side'
        raise AttributeError(f"'OrderIB' object has no attribute '{name}'")
