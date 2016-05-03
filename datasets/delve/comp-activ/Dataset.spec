# Title: Computer system activity. Measures of a of system activity in a multi-processor, multiuser computer system.
# Origin: cultivated
# 
# Usage: development
# 
# Order: uninformative
# 
# Attributes:
  1  time      i     ?		# Time of day
  2  lread     u  0..Inf	# Number of system buffer reads per second
  3  lwrite    u  0..Inf	# Number of system buffer writes per second
  4  scall     u  0..Inf	# Number of system calls per second
  5  sread     u  0..Inf	# Number of system call reads per second
  6  swrite    u  0..Inf	# Number of system call writes per second
  7  fork      u  0..Inf	# Number of system call forks per second
  8  exec      u  0..Inf	# Number of system call execs per second
  9  rchar     u  0..Inf	# Number of characters transferred by sys reads
 10  wchar     u  0..Inf	# Number of characters transferred by sys writes
 11  pgout     u  [0,Inf]	# Page-out requests per second
 12  ppgout    u  [0,Inf]	# Pages paged out per second
 13  pgfree    u  [0,Inf]	# Pages freed per second
 14  pgscan    u  [0,Inf]	# Pages scanned for freeing per second
 15  atch      u  [0,Inf]	# Page faults/sec satisfied by attaches
 16  pgin      u  [0,Inf]	# Page in requests per second
 17  ppgin     u  [0,Inf]	# Pages paged in per second
 18  pflt      u  [0,Inf]	# "Copy-on-write" page faults
 19  vflt      u  [0,Inf]	# Page faults caused by pages not in memory
 20  runqsz    u  [0,Inf]	# Process run queue size
 21  runocc    u  0..100	# % time that that run queue is occupied
 22  freemem   u  0..512	# Number of free memory pages
 23  freeswap  u  0..15000	# Free disk swap blocks
 24  usr       y  0..100	# % CPU utilization (user)
 25  sys       u  0..100	# % CPU utilization (system)
 26  wio       u  0..100	# % CPU utilization (Idle and waiting for I/O)
 27  idle      u  0..100	# % CPU utilization (Idle)
