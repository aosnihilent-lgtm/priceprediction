from pydantic import BaseModel
from datetime import datetime

class QuoteRequest(BaseModel):
    SOLDTO: str
    MATERIALNO: str
    SALESORG: str
    ZIPCODE: str
    PERUNITCOST: float
    STDINVOICEPRICE: float
    PERUNITQUOTEPRICE: float
    QUANTITY: int
    CURRENCY: str
    CREATEDON: datetime
    SALESOFFICE: str
    PRODUCTHIERARCHY : str