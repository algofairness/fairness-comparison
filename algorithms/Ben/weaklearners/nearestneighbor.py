def dist(x,y):
   return sum((a-b)**2 for (a,b) in zip(x,y))


def nearestLearner(draw):
   data = [draw() for _ in range(100)]

   def classify(x):
      return min(data, key=lambda y: dist(x, y[0]))[1]

   return classify
