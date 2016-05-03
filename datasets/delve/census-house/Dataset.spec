#
# Dataset.spec
#
#Origin: cultivated
#Usage: assessment
#Order: uninformative
#Attributes:
#
 1	State	c 0..55	# State numbers
 2	Code	i [0,+Inf]	# unique Place code
 3	P1	u [1,+Inf]	# total persons
 4	P2	u [0,+Inf]	# total families
 5	P3	u [0,+Inf]	# total HH-olds
 6	P4.1	u [0,1]		# %-tage living inside urbanized area
 7	P4.2	u [0,1]		# %-tage living outside urbanized area (but still urban)
 8	P4.3	u [0,1]		# %-tage living inside rural 
 9	P4.4	u [0,1]		# %-tage living elsewhere
 10	P5.1	u [0,1]		# %-tage male
 11	P5.2	u [0,1]		# %-tage female
 12	P6.1	u [0,1]		# %-tage white
 13	P6.2	u [0,1]		# %=tage black
 14	P6.3	u [0,1]		# %-tage American Indian, Eskimo or Aleut (indian)
 15	P6.4	u [0,1]		# %-tage Asian or Pacific Inslander (asian)
 16	P6.5	u [0,1]		# %-tage other race
 17	P8	u [0,1]		# %-tage of hispanic origin (hispanic)
 18	P11.1	u [0,1]		# %-tage (0:11] years old	
 19	P11.2	u [0,1]		# %-tage [12:24] years old	
 20	P11.3	u [0,1]		# %-tage [25:64] years old	
 21	P11.4	u [0,1]		# %-tage [65:\infty] years old	
 22	P14.1	u [0,1]		# %-tage males never married
 23	P14.2	u [0,1]		# %-tage males  married, not separated
 24	P14.3	u [0,1]		# %-tage males separated
 25	P14.4	u [0,1]		# %-tage males widowed
 26	P14.5	u [0,1]		# %-tage males divorced
 27	P14.6	u [0,1]		# %-tage females never married
 28	P14.7	u [0,1]		# %-tage females  married, not separated
 29	P14.8	u [0,1]		# %-tage females separated
 30	P14.9	u [0,1]		# %-tage females widowed
 31	P14.10	u [0,1]		# %-tage females divorced
 32	P15.1	u [0,1]		# %-tage of people in family HH-s
 33	P15.2	u [0,1]		# %-tage of people in non-family HH-s
 34	P15.3	u [0,1]		# %-tage of people in group quarters (incl jails)
 35	P16.1	u [0,1]		# %-tage of HH-lds with 1 person
 36	P16.2	u [0,1]		# %-tage of HH-lds with 2 or more persons which are family HH-lds
 37	P16.3	u [0,1]		# %-tage of HH-lds which are non-family
 38	P17	u [0,1]		# %-tage of peoples in families
 39	P17a	u [0,+Inf]	# average family size
 40	P18.1	u [0,1]		# %-tage of HH-lds with 1+ persons under 18 which are family HH-lds
 41	P18.2	u [0,1]		# %-tage of HH-lds with 1+ persons under 18 which are non-family HH-lds
 42	P18.3	u [0,1]		# %-tage of HH-lds with no persons under 18 which are family HH-lds
 43	P18.4	u [0,1]		# %-tage of HH-lds with no persons under 18 which are non-family HH-lds
 44	P19.1	u [0,1]		# %-tage of HH-lds with white HH-lder
 45	P19.2	u [0,1]		# %-tage of HH-lds with black HH-lder
 46	P19.3	u [0,1]		# %-tage of HH-lds with indian HH-lder
 47	P19.4	u [0,1]		# %-tage of HH-lds with asian HH-lder
 48	P19.5	u [0,1]		# %-tage of HH-lds with other HH-lder
 49	P20.1	u [0,1]		# %-tage of HH-lds  with Hispanic HH-lder
 50	P25.1	u [0,1]		# %-tage of HH-lds with 1+ persons 65 and over
 51	P25.2	u [0,1]		# %-tage of HH-lds with no persons 65 and over
 52	P26.1	u [0,1]		# %-tage of HH-lds with 1+ non-relatives
 53	P26.2	u [0,1]		# %-tage of HH-lds with no non-relatives
 54	P27.1	u [0,1]		# %-tage of HH-lds which are small family HH-lds [2-4] persons
 55	P27.2	u [0,1]		# %-tage of HH-lds which are large family HH-lds 5+ persons
 56	P27.3	u [0,1]		# %-tage of HH-lds which have one person
 57	P27.4	u [0,1]		# %-tage of HH-lds which are non-family with 2+ persons
 58	H1	u [0,+Inf]	# Total num of Housing Units(HU)
 59	H2.1	u [0,1]		# %-tage of HU occupied
 60	H2.2	u [0,1]		# %-tage of HU vacant
 61	H3.1	u [0,1]		# %-tage of occupied HU owner occupied (ownOcc)
 62	H3.2	u [0,1]		# %-tage of occupied HU renter occ-ed (rentOcc)
 63	H4.1	u [0,1]		# %-tage of HU inside urban area
 64	H4.2	u [0,1]		# %-tage of HU outside urban area (but still within Urban)
 65	H4.3	u [0,1]		# %-tage of HU in rural area
 66	H4.4	u [0,1]		# %-tage of HU in other areas
 67	H5.1	u [0,1]		# %-tage of vacant HU for rent
 68	H5.2	u [0,1]		# %-tage of vacant HU for sale only
 69	H5.3	u [0,1]		# %-tage of vacant HU rented, sold not occ-ed
 70	H5.4	u [0,1]		# %-tage of vacant HU for seasonal, rec or occasional use
 71	H5.5	u [0,1]		# %-tage of vacant HU for migrant workers
 72	H5.6	u [0,1]		# %-tage of vacant HU other
 73	H7.1	u [0,1]		# %-tage of vacant HU with usual home elsewhere
 74	H8.1	u [0,1]		# %-tage of occ-ed HU with white HH-lder
 75	H8.2	u [0,1]		# %-tage of occ-ed HU with black HH-lder
 76	H8.3	u [0,1]		# %-tage of occ-ed HU with indian HH-lder
 77	H8.4	u [0,1]		# %-tage of occ-ed HU with asian HH-lder
 78	H8.5	u [0,1]		# %-tage of occ-ed HU with other HH-lder
 79	H9.1	u [0,1]		# %-tage of ownOcc HU with white HH-lder
 80	H9.2	u [0,1]		# %-tage of ownOcc HU with black HH-lder
 81	H9.3	u [0,1]		# %-tage of ownOcc HU with indian HH-lder
 82	H9.4	u [0,1]		# %-tage of ownOcc HU with asian HH-lder
 83	H9.5	u [0,1]		# %-tage of ownOcc HU with other HH-lder
 84	H9.6	u [0,1]		# %-tage of rentOcc HU with white HH-lder
 85	H9.7	u [0,1]		# %-tage of rentOcc HU with black HH-lder
 86	H9.8	u [0,1]		# %-tage of rentOcc HU with indian HH-lder
 87	H9.9	u [0,1]		# %-tage of rentOcc HU with asian HH-lder
 88	H9.10	u [0,1]		# %-tage of rentOcc HU with other HH-lder
 89	H10.1	u [0,1]		# %-tage of occ-ed HU with HH-lder not of Hispanic origin
 90	H10.2	u [0,1]		# %-tage of occ-ed HU with HH-lder of Hispanic origin
 91	H12.1	u [0,+Inf]	# Average age of HH-lder in ownOcc HU's
 92	H12.2	u [0,+Inf]	# Average age of HH-lder in rentOcc HU's
 93	H13.1	u [0,1]		# %-tage of HU with 1-4 rooms
 94	H13.2	u [0,1]		# %-tage of HU with 5+ rooms
 95	H14	u [0,+Inf] 	# Average number of rooms in a HU
 96	H15.1	u [0,+Inf]	# Average number of rooms in a ownOcc HU
 97	H15.2	u [0,+Inf]	# Average number of rooms in a rentOcc HU
 98	H17.1	u [0,1]		# %-tage of occ-ed HU with 1-4 persons
 99	H17.2	u [0,1]		# %-tage of occ-ed HU with 5+ persons
 100	H18.1	u [0,1]		# %-tage of ownOccHU with 1-4 persons
 101	H18.2	u [0,1]		# %-tage of ownOcc HU with 5+ persons
 102	H18.3	u [0,1]		# %-tage of rentOcc HU with 1-4 persons
 103	H18.4	u [0,1]		# %-tage of rentOcc HU with 5+ persons
 104	H18.A	u [0,+Inf]	# average number of persons per ownOcc HU
 105	H18.A	u [0,+Inf]	# average number of persons per rentOcc HU
 106	H19	u [0,+Inf]	# average num of persons in HU
 107	H23.1	u [0,1]		# %-tage of (specified) ownOcc HU with value (0-50,000)
 108	H23.2	u [0,1]		# %-tage of (specified) ownOcc HU with value [50,000-100,000)
 109	H23.3	u [0,1]		# %-tage of (specified) ownOcc HU with value (100,000-250,000)
 110	H23.4	u [0,1]		# %-tage of (specified) ownOcc HU with value >=250,000
 111	H23.A	u [0,+Inf]	# lower quartile of HU price
 112	H23.B   y [0,+Inf]	# median HU price
 113	H23.C	u [0,+Inf]	# upper quartile of HU price
 114	H24	u [0,+Inf]	# average HU price.
 115	H26.1	u [0,+Inf]	# average value of (specified) ownOcc HU with white owner
 116	H26.2	u [0,+Inf]	# average value of (specified) ownOcc HU with black owner
 117	H26.3	u [0,+Inf]	# average value of (specified) ownOcc HU with indian owner
 118	H26.4	u [0,+Inf]	# average value of (specified) ownOcc HU with asian owner
 119	H26.5	u [0,+Inf]	# average value of (specified) ownOcc HU with other owner
 120	H28.1	u [0,+Inf]	# average value of (specified) ownOcc HU with non-hispanic owner
 121	H28.2	u [0,+Inf]	# average value of (specified) ownOcc HU with hispanic owner
 122	H31	u [0,+Inf]	# average price asked for HU
 123	H32.A	u [0,+Inf]	# lower quartile of HU rent
 124	H32.B	u [0,+Inf]	# median HU rent
 125	H32.C	u [0,+Inf]	# upper quartile of HU rent
 126	H33	u [0,+Inf]	# average HU rent.
 127	H35.1	u [0,+Inf]	# Average rent for HU with white HH-lder
 128	H35.2	u [0,+Inf]	# Average rent for HU with black HH-lder
 129	H35.3	u [0,+Inf]	# Average rent for HU with indian HH-lder
 130	H35.4	u [0,+Inf]	# Average rent for HU with asian HH-lder
 131	H35.5	u [0,+Inf]	# Average rent for HU with other HH-lder
 132	H37.1	u [0,+Inf]	# Average rent for HU with non-hispanic HH-lder
 133	H37.2	u [0,+Inf]	# Average rent for HU with hispanic HH-lder
 134	H38	u [0,+Inf]	# Average rent asked for vacant-for-rent HU's
 135	H40.1	u [0,1]		# %-tage of vacant-for-rent HU vacant less then 2 months
 136	H40.2	u [0,1]		# %-tage of vacant-for-rent HU vacant more then 6 months
 137	H40.3	u [0,1]		# %-tage of vacant-for-sale HU vacant less then 2 months
 138	H40.4	u [0,1]		# %-tage of vacant-for-sale HU vacant more then 6 months
