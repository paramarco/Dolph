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
        order = self._raw.order
        if order.orderType in ('STP', 'STP LMT'):
            if hasattr(order, 'auxPrice') and order.auxPrice and order.auxPrice < self._UNSET_DOUBLE:
                price = order.auxPrice
        if price is None and hasattr(order, 'lmtPrice') and order.lmtPrice and order.lmtPrice < self._UNSET_DOUBLE:
            price = order.lmtPrice
        if price is None and hasattr(self._raw.orderStatus, 'avgFillPrice') and self._raw.orderStatus.avgFillPrice:
            price = self._raw.orderStatus.avgFillPrice
        
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
