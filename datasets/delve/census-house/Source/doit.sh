#! /bin/sh

for i in *.dat.1.gz
do
	echo -n "$i ... " > /dev/tty	
	zcat $i|gawk -f choose.awk 
	echo "done" > /dev/tty
done | gzip > data.done.gz
