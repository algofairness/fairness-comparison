from data.objects.Sample import Sample
from data.objects.Ricci import Ricci
from data.objects.Adult import Adult
from data.objects.German import German
from data.objects.PropublicaRecidivism import PropublicaRecidivism
from data.objects.PropublicaViolentRecidivism import PropublicaViolentRecidivism
from data.objects.TwoGaussians import TwoGaussians

DATASETS = [
    TwoGaussians(),
    Ricci(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    PropublicaViolentRecidivism()
    ]

# For testing, you can just use a sample of the data.  E.g.:
# DATASETS = [ Sample(Adult(), 50) ]
# DATASETS = [Sample(d, 10) for d in DATASETS]

def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names

