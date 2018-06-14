from fairness.data.objects.Sample import Sample
from fairness.data.objects.Ricci import Ricci
from fairness.data.objects.Adult import Adult
from fairness.data.objects.German import German
from fairness.data.objects.PropublicaRecidivism import PropublicaRecidivism
from fairness.data.objects.PropublicaViolentRecidivism import PropublicaViolentRecidivism
from fairness.data.objects.TwoGaussians import TwoGaussians

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

