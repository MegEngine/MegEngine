#! /bin/bash -e
set -e

if [ $# -lt 2 ] ; then
echo "USAGE: $0 src dst"
echo " e.g.: $0 ~/xxx.mdl ~/xxx.encrypted.mdl"
echo " e.g.: $0 ~/xxx.mdl ~/xxx.encrypted.mdl key"
exit 1;
fi

IV=`openssl rand -hex 16`

Key=000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F
if [ $# == 3 ] ; then
Key=$3
fi

# get file size
size=`wc -c $1`

echo "encrypt aes-256-cbc ..."
openssl enc -e -aes-256-cbc -in $1 -out $1.tmp -K $Key -iv $IV
echo $IV | xxd -r -p | cat - $1.tmp > $2 
# write size into file
printf "%016x" ${size%\ *} | xxd -r -p >> $2
rm -f $1.tmp
