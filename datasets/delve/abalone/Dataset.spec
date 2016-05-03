#
# Abalone
#   Predicting the age of abalone from physical measurements.  The age of
#   abalone is determined by cutting the shell through the cone, staining it,
#   and counting the number of rings through a microscope -- a boring and
#   time-consuming task.  Other measurements, which are easier to obtain, are
#   used to predict the age.  Further information, such as weather patterns
#   and location (hence food availability) may be required to solve the problem.
#
#   From the original data examples with missing values were removed (the
#   majority having the predicted value missing), and the ranges of the
#   continuous values have been scaled for use with an ANN (by dividing by 200).
#
#   Data comes from an original (non-machine-learning) study:
#
#	Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and
#	Wes B Ford (1994) "The Population Biology of Abalone (_Haliotis_
#	species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the North
#	Coast and Islands of Bass Strait", Sea Fisheries Division, Technical
#	Report No. 48 (ISSN 1034-3288)
#
#Title: UCI Abalone Database. Predict number of rings (age) of abalone from physical measurements
#Origin: natural
#
#Usage: assessment
#
#Order: uninformative
#
# Attributes:
  1   sex                 c  M,F,I	# Gender or Infant (I)
  2   length              u  (0,Inf]	# Longest shell measurement (mm)
  3   diameter            u  (0,Inf]	# perpendicular to length     (mm)
  4   height              u  (0,Inf]	# with meat in shell (mm)
  5   whole_weight        u  (0,Inf]	# whole abalone  (gr)
  6   shucked_weight      u  (0,Inf]	# weight of meat (gr)    
  7   viscera_weight      u  (0,Inf]	# gut weight (after bleeding) (gr)
  8   shell_weight        u  (0,Inf]	# after being dried (gr)
  9   rings               y  0..29	# +1.5 gives the age in years
