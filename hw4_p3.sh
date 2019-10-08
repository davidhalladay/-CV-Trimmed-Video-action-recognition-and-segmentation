# TODO: create shell script for Problem 3

wget -O s2s-RNN043-085-0.534085.pth https://www.dropbox.com/s/s96xvrv798nabfw/s2s-RNN043-085-0.534085.pth?dl=1

python3 ./p3_valid_reproduce.py s2s-RNN043-085-0.534085.pth $1 $2 
