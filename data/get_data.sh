#! /bin/bash

DATADIR="$( cd ../"$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
wget https://www.dropbox.com/s/2nsklcbm4hkiuvf/proSR-data.zip
unzip proSR-data.zip $DATADIR
rm -rf proSR-data.zip

