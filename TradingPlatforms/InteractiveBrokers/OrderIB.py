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

    def __str__(self):
        """
        String representation for debugging.
        """
        msg = f"OrderIB(id={self.id}, symbol={self.symbol}, side={self.side}, "
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
