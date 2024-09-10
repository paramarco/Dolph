from alpaca_trade_api.entity import Order as AlpacaOrder  # Assuming this is the module where Alpaca's Order is defined

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
        return f"CustomOrder(id={self.id}, symbol={self.symbol}, status={self.status})"
   
    def __getattr__(self, name):
        """
        Custom attribute handler. Maps 'seccode' to 'symbol' and defers to the superclass for other attributes.
        """
        if name == 'seccode':
            return self.symbol  # Map 'seccode' to 'symbol'
        if name == 'buysell':
            return self.side  # Map 'buysell' to 'side'
        try:
            # If the attribute is not 'seccode', defer to the superclass (AlpacaOrder)
            return super().__getattr__(name)
        
        except AttributeError:
            raise AttributeError(f"'OrderAlpaca' object has no attribute '{name}'")

