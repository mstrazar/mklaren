#
# Puma forward dynamics -- 8fm = 8 inputs, low nonlinearity, medium noise
#
#
#Origin: simulated
#
#Usage: assessment
#
#Order: uninformative
#
#Attributes:
  1  theta1	      u  [-3.1416,3.1416]	# ang position of joint 1 in radians
  2  theta2	      u  [-3.1416,3.1416]	# ang position of joint 2 in radians
  3  theta3	      u  [-3.1416,3.1416]	# ang position of joint 3 in radians
  4  thetad1      u  (-Inf,Inf)	# ang vel of joint 1 in rad/sec
  5  thetad2      u  (-Inf,Inf)	# ang vel of joint 2 in rad/sec
  6  thetad3      u  (-Inf,Inf)	# ang vel of joint 3 in rad/sec
  7  tau1     u  (-Inf,Inf)	# torque on jt 1 
  8  tau2     u  (-Inf,Inf)	# torque on jt 2
  9  thetadd3     y  (-Inf,Inf)	# ang acceleration of joint 3
