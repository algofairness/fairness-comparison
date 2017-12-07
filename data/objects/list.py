from Ricci import Ricci
from Adult import Adult
from German import German
from Retailer import Retailer

DATASETS = [ Ricci(), Adult(), German(), Retailer() ]

def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names

