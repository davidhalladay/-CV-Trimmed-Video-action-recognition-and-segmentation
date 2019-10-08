# TODO: create shell script for running your DANN model

wget -O usps_F.pth https://www.dropbox.com/s/ylvxb5xcbeftjjh/DANN-F-usps.pth?dl=1
wget -O usps_L.pth https://www.dropbox.com/s/nr953fb1tv6tgpg/DANN-L-usps.pth?dl=1
wget -O svhn_F.pth https://www.dropbox.com/s/mzq6udj7jwxw1e3/DANN-F-svhn.pth?dl=1
wget -O svhn_L.pth https://www.dropbox.com/s/qu7333s98et0ety/DANN-L-svhn.pth?dl=1
wget -O mnistm_F.pth https://www.dropbox.com/s/anublyv229x33v0/DANN-F-mnistm.pth?dl=1
wget -O mnistm_L.pth https://www.dropbox.com/s/bd2m95hlj7kqtr6/DANN-L-mnistm.pth?dl=1

if [ "$2" = "mnistm" ]
then
    python3 ./DANN_predict.py $1 $3 usps_F.pth usps_L.pth
elif [ "$2" = "usps" ]
then
    python3 ./DANN_predict.py $1 $3 svhn_F.pth svhn_L.pth
elif [ "$2" = "svhn" ]
then
     python3 ./DANN_predict.py $1 $3 mnistm_F.pth mnistm_L.pth
fi
