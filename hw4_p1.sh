# TODO: create shell script for Problem 1

wget -O VCNN-018.pth https://www.dropbox.com/s/i4jvva3z0yexhzi/CNN-018.pth?dl=1

python3 ./p1_valid_reproduce.py VCNN-018.pth $1 $2 $3
