#
# boston dataset
#
#
#Origin: natural
#
#Usage: development
#
#Order: ?
#
#Attributes:
  1  CRIM     u  [0,Inf)    # per capita crime rate by town 
  2  ZN       u  [0,100]    # proportion of residential land zoned for lots over 25,000 sq.ft.
  3  NDUS     u  [0,100]    # proportion of non-retail business acres per town
  4  CHAS     u  0 1        # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
  5  NOX      u  [0,Inf)    # nitric oxides concentration (parts per 10 million)
  6  RM       u  (0,Inf)    # average number of rooms per dwelling
  7  AGE      u  [0,100]    # proportion of owner-occupied units built prior to 1940
  8  DIS      u  (0,Inf)    # weighted distances to five Boston employment centres
  9  RAD      u  (0..Inf)   # index of accessibility to radial highways
 10  TAX      u  [0,Inf)    # full-value property-tax rate per $10,000
 11  PTRATIO  u  (0,Inf)    # pupil-teacher ratio by town
 12  B        u  [0,396.9]  # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 13  LSTAT    u  [0,100]    # % lower status of the population
 14  MEDV     y  (0,Inf)    # Median value of owner-occupied homes in $1000's
