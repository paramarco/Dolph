from alpaca_trade_api.entity import Order as AlpacaOrder  # Assuming this is the module where Alpaca's Order is defined
from datetime import datetime, timezone
import pandas as pd

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
            created_at = getattr(self, "created_at", None)  # Obtener 'created_at' si existe

            if created_at:
                try:
                    # Si es un string, convertirlo usando fromisoformat
                    if isinstance(created_at, str):
                        return datetime.fromisoformat(created_at.rstrip('Z')).replace(tzinfo=timezone.utc)
                    
                    # Si es un objeto Timestamp (por ejemplo, de Pandas), convertirlo a datetime
                    elif isinstance(created_at, pd.Timestamp):
                        return created_at.to_pydatetime().replace(tzinfo=timezone.utc)
                    
                    # Si ya es un objeto datetime, devolverlo directamente
                    elif isinstance(created_at, datetime):
                        return created_at
                    
                    else:
                        raise TypeError(f"Unexpected type for 'created_at': {type(created_at)}")
                
                except (ValueError, TypeError) as e:
                    raise AttributeError(f"'OrderAlpaca' object has an invalid 'created_at' format: {created_at} ({e})")

            else:
                raise AttributeError("'OrderAlpaca' object has no valid 'created_at' attribute")





        try:
            # If the attribute is not 'seccode', defer to the superclass (AlpacaOrder)
            return super().__getattr__(name)
        
        except AttributeError:
            raise AttributeError(f"'OrderAlpaca' object has no attribute '{name}'")

