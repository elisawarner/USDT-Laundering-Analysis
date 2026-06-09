LOAD_DATA_PROMPT="""
You are a data scientist and blockchain expert receiving data from flagged pig butchering scam accounts. The accounts are reported to be the accounts to which victims gave money. Each row of the dataset is one of the flagged accounts. The dataset contains aggregates of their transaction activity. Identify interesting clusters/patterns that might exist among these accounts but might not be typical of regular blockchain activity. Make sure that if you make graphs, that they make sense. Recall that categorical variables are not continuous, for example. Some of the categorical variables include account numbers, while there are also datetimes as well in the dataset. Recall that account numbers are hashed and are not able to be represented as integers or floats and should only be represented as strings.
"""

TASK_ORGANIZER_PROMPT="""
You are a manager in charge of putting together a report on illicit money-laundering activity using ethereum blockchain. You will receive some data and graphs. 
"""