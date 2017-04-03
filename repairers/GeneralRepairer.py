from AbstractRepairer import AbstractRepairer
from CategoricRepairer import Repairer as CategoricRepairer
from NumericRepairer import Repairer as NumericRepairer

class Repairer(AbstractRepairer):
  def __init__(self, *args, **kwargs):
    super(Repairer, self).__init__(*args, **kwargs)

    if all(isinstance(row[self.feature_to_repair],float) for row in self.all_data):
      self.repairer = NumericRepairer(*args, **kwargs)
    elif all(isinstance(row[self.feature_to_repair],int) for row in self.all_data):
      self.repairer = NumericRepairer(*args, **kwargs)
    elif all(isinstance(row[self.feature_to_repair],str) for row in self.all_data):
      self.repairer = CategoricRepairer(*args, **kwargs)

  def repair(self, data_to_repair):
    return self.repairer.repair(data_to_repair)
