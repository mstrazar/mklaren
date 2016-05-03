#
# Forward kinematics of an 8 link robot arm -- 32fm = 32 inputs, low
# nonlinearity, med noise 
#
#
#Origin: simulated
#
#Usage: development
#
#Order: uninformative
#
#Attributes:
  1  theta1	      u  [-3.1416,3.1416]	# ang position of joint 1 in radians
  2  theta2	      u  [-3.1416,3.1416]	# ang position of joint 2 in radians
  3  theta3	      u  [-3.1416,3.1416]	# ang position of joint 3 in radians
  4  theta4	      u  [-3.1416,3.1416]	# ang position of joint 4 in radians
  5  theta5	      u  [-3.1416,3.1416]	# ang position of joint 5 in radians
  6  theta6	      u  [-3.1416,3.1416]	# ang position of joint 6 in radians
  7  theta7	      u  [-3.1416,3.1416]	# ang position of joint 7 in radians
  8  theta8	      u  [-3.1416,3.1416]	# ang position of joint 8 in radians
  9  alpha1	      u  [-3.1416,3.1416]	# link 1 twist angle
  10 alpha2	      u  [-3.1416,3.1416]	# link 2 twist angle
  11 alpha3	      u  [-3.1416,3.1416]	# link 3 twist angle
  12 alpha4	      u  [-3.1416,3.1416]	# link 4 twist angle
  13 alpha5	      u  [-3.1416,3.1416]	# link 5 twist angle
  14 alpha6	      u  [-3.1416,3.1416]	# link 6 twist angle
  15 alpha7	      u  [-3.1416,3.1416]	# link 7 twist angle
  16 alpha8	      u  [-3.1416,3.1416]	# link 8 twist angle
  17 a1	      u  [0, Inf)		# link 1 length
  18 a2	      u  [0, Inf)		# link 2 length
  19 a3	      u  [0, Inf)		# link 3 length
  20 a4	      u  [0, Inf)		# link 4 length
  21 a5	      u  [0, Inf)		# link 5 length
  22 a6	      u  [0, Inf)		# link 6 length
  23 a7	      u  [0, Inf)		# link 7 length
  24 a8	      u  [0, Inf)		# link 8 length
  25 d1	      u  [0, Inf)		# link 1 offset distance
  26 d2	      u  [0, Inf)		# link 2 offset distance
  27 d3	      u  [0, Inf)		# link 3 offset distance
  28 d4	      u  [0, Inf)		# link 4 offset distance
  29 d5	      u  [0, Inf)		# link 5 offset distance
  30 d6	      u  [0, Inf)		# link 6 offset distance
  31 d7	      u  [0, Inf)		# link 7 offset distance
  32 d8	      u  [0, Inf)		# link 8 offset distance
  33 y	      y  [0, Inf)  # Cartesian distance of end point from position (0.1, 0.1, 0.1)
