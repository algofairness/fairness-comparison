Kamishima and Zafar seem to discretize data differently.

!!!!!!!!!!!!!!!!!
In Kamishima data, the second to last value is the "sensitive feature" (gender), and the last value is the class to be predicted (income)
!!!!!!!!!!!!!!!!!
For the rest of the discretization, see the Calder's paper, in particular they do this:
  Technique for discretizing data in case of Kamishima
  "In addition, we remove low
  frequency counts (which may lead to problems for EM) by pooling any bin that
  occurs less than 50 times (out of a total of about 16 000). Thus, all infrequent
  attribute values are replaced by a unique (more frequent) ‘pool’ value. On this
  modified data-set we tested our algorithms."

In Zafar's discretization we are given beforehand an X, y and x_control,
  where X is the data, y is the value to be predicted (equivalent to the last value
  in Kamishima's discretization) and x_control is gender status
