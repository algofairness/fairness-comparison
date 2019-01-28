from fairness.data.objects.Sample import Sample
from fairness.data.objects.Ricci import Ricci
from fairness.data.objects.Adult import Adult
from fairness.data.objects.German import German
from fairness.data.objects.PropublicaRecidivism import PropublicaRecidivism
from fairness.data.objects.PropublicaViolentRecidivism import PropublicaViolentRecidivism
from fairness.data.objects.TwoGaussians import TwoGaussians

DATASETS = [

# Synthetic datasets to test effects of class balance:
#   TwoGaussians(0.1), TwoGaussians(0.2), TwoGaussians(0.3), TwoGaussians(0.4),
#   TwoGaussians(0.5), TwoGaussians(0.6), TwoGaussians(0.7), TwoGaussians(0.8), TwoGaussians(0.9),

# Downsampled datasetes to test effects of class and protected class balance:
#     Sample(Ricci(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5, sensitive_attr="Race"),
#     Sample(Adult(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
#     sensitive_attr="race-sex"),
#     Sample(German(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5, sensitive_attr="sex-age"),
#     Sample(PropublicaRecidivism(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
#     sensitive_attr="sex-race"),
#     Sample(PropublicaViolentRecidivism(), num = 1000, prob_pos_class = 0.5, prob_privileged = 0.5,
#     sensitive_attr="sex-race")

# Real datasets:
    Ricci(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    PropublicaViolentRecidivism()
    ]


def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names

def add_dataset(dataset):
    DATASETS.append(dataset)

def get_dataset_by_name(name):
    for ds in DATASETS:
        if ds.get_dataset_name() == name:
            return ds
    raise Exception("No dataset with name %s could be found." % name)
    
