from pydantic import ValidationError
import json
from .schemas import TicketOutput
def parse_and_validate_ticket(raw_output:str) : 
    try : 
        data = json.loads(raw_output)
        ticket = TicketOutput.model_validate(data)
        return ticket

    except json.JSONDecodeError :
        raise ValueError("Model output is not valid JSON")

    except ValidationError : 
        raise ValueError ('MODEL output does not match ticket')
