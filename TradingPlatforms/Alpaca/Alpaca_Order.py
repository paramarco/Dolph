from alpaca_trade_api.entity import Order as AlpacaOrder  # Assuming this is the module where Alpaca's Order is defined
from datetime import datetime, timezone

class OrderAlpaca(AlpacaOrder):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __eq__(self, other):
        """
        Compare orders based on their 'id' attribute.
        """
        if isinstance(other, OrderAlpaca):
            return self.id == other.id
        return False
    
    def __str__(self):
        """
        String representation for debugging.
        """
        msg = f"CustomOrder(id={self.id}, symbol={self.symbol}, side={self.side}, "
        msg += f"status={self.status}, time={self.time.isoformat()})"
        return msg
   
    def __getattr__(self, name):
        """
        Custom attribute handler. Maps 'seccode' to 'symbol' and defers to the superclass for other attributes.
        """
        if name == 'seccode':
            return self.symbol  # Map 'seccode' to 'symbol'
        if name == 'buysell':
            return self.side  # Map 'buysell' to 'side'
        if name == 'time':
            created_at_str = getattr(self, "created_at", None)  # Obtener 'created_at' si existe

            if created_at_str:
                try:
                    # Convertir 'created_at' en un datetime UTC
                    return datetime.fromisoformat(created_at_str.rstrip('Z')).replace(tzinfo=timezone.utc)
                except ValueError:
                    raise AttributeError(f"'OrderAlpaca' object has invalid 'created_at' format: {created_at_str}")
            else:
                raise AttributeError("'OrderAlpaca' object has no valid 'created_at' attribute")



        try:
            # If the attribute is not 'seccode', defer to the superclass (AlpacaOrder)
            return super().__getattr__(name)
        
        except AttributeError:
            raise AttributeError(f"'OrderAlpaca' object has no attribute '{name}'")

