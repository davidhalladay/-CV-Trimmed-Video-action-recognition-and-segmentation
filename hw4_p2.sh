# TODO: create shell script for Problem 2

wget -O RNN-043.pth https://www.dropbox.com/s/bq5oy152ettx6q2/RNN-043.pth?dl=1

python3 ./p2_valid_reproduce.py RNN-043.pth $1 $2 $3
