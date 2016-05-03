#
# bank-32nh dataset
#
#
#Origin: simulated
#
#Usage: assesment
#
#Order: ?
#
#Attributes:
  1  a1cx     u  (-1.0,1.0) # x coordinate of centre of residential area 1
  2  a1cy     u  (-1.0,1.0) # y coordinate of centre of residential area 1
  3  a1sx     u  (0,Inf)    # spread of residential area 1 in x direction
  4  a1sy     u  (0,Inf)    # spread of residential area 1 in y direction
  5  a1rho    u  (-0.5,0.5) # correlation for residental distribution 1
  6  a1pop    u  (0,Inf)    # population size for area 1
  7  a2cx     u  (-1.0,1.0) # x coordinate of centre of residential area 2
  8  a2cy     u  (-1.0,1.0) # y coordinate of centre of residential area 2
  9  a2sx     u  (0,Inf)    # spread of residential area 2 in x direction
 10  a2sy     u  (0,Inf)    # spread of residential area 2 in y direction
 11  a2rho    u  (-0.5,0.5) # correlation for residental distribution 2
 12  a2pop    u  (0,Inf)    # population size for area 2
 13  a3cx     u  (-1.0,1.0) # x coordinate of centre of residential area 3
 14  a3cy     u  (-1.0,1.0) # y coordinate of centre of residential area 3
 15  a3sx     u  (0,Inf)    # spread of residential area 3 in x direction
 16  a3sy     u  (0,Inf)    # spread of residential area 3 in y direction
 17  a3rho    u  (-0.5,0.5) # correlation for residental distribution 3
 18  a3pop    u  (0,Inf)    # population size for area 3
 19  temp     u  (0,1)      # ficticious temperature controlling bank choice
 20  b1x      u  (-1.0,1.0) # x coordinate of location of bank 1
 21  b1y      u  (-1.0,1.0) # y coordinate of location of bank 1
 22  b1call   u  2..8       # queues are opened if other ones exceed length  
 23  b1eff    u  (0.5,2.0)  # efficiency of bank number 1
 24  b2x      u  (-1.0,1.0) # x coordinate of location of bank 2
 25  b2y      u  (-1.0,1.0) # y coordinate of location of bank 2
 26  b2call   u  2..8       # queues are opened if other ones exceed length  
 27  b2eff    u  (0.5,2.0)  # efficiency of bank number 2
 28  b3x      u  (-1.0,1.0) # x coordinate of location of bank 3
 29  b3y      u  (-1.0,1.0) # y coordinate of location of bank 3
 30  b3call   u  2..8       # queues are opened if other ones exceed length  
 31  b3eff    u  (0.5,2.0)  # efficiency of bank number 3
 32  mxql     u  5..9       # maximum possible length of queues
 33  rej      y  (0,1)      # rejection rate (fraction)
