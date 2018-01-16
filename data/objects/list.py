from data.objects.Ricci import Ricci
from data.objects.Adult import Adult
from data.objects.German import German
from data.objects.PropublicaRecidivism import PropublicaRecidivism

DATASETS = [ Ricci(), Adult(), German(), PropublicaRecidivism() ]

def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names

