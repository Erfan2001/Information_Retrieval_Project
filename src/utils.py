from src.tools.config import pars_args
# import timespan
from datetime import datetime


def span_time():
    business_hours = [
        "9:00-17:00|mon-fri|*|*",  # is between 9 a.m. to 5 p.m. on Mon to Fri
        "!*|*|1|jan",  # not new years
        "!*|*|25|dec",  # not christmas
        "!*|thu|22-28|nov",  # not thanksgiving
    ]
    if timespan.match(business_hours, datetime.now()):
        print("The model can be trained!")
    else:
        print("The model cannot be trained! (Off Day)")


def load_config_file():
    args = pars_args()
    return args
