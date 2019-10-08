# TODO: create shell script for running your improved UDA model

wget -O usps_F_best.pth https://www.dropbox.com/s/ddeb8mwxbc45qd6/DANN-F-usps-best.pth?dl=1
wget -O usps_L_best.pth https://www.dropbox.com/s/thy1rui59apbgs9/DANN-L-usps-best.pth?dl=1
wget -O DSN.pth https://www.dropbox.com/s/ypntzbsptzgmg3b/DSN-svhn.pth?dl=1
wget -O mnistm_F_best.pth https://www.dropbox.com/s/o5mnhw8jz2ghb5y/DANN-F-mnistm-best.pth?dl=1
wget -O mnistm_L_best.pth https://www.dropbox.com/s/9c56glmkt7hzmcs/DANN-L-mnistm-best.pth?dl=1

if [ "$2" = "mnistm" ]
then
    python3 ./DANN_predict_imp.py $1 $3 usps_F_best.pth usps_L_best.pth
elif [ "$2" = "usps" ]
then
    python3 ./DSN_predict.py $1 $3 DSN.pth
elif [ "$2" = "svhn" ]
then
     python3 ./DANN_predict_imp.py $1 $3 mnistm_F_best.pth mnistm_L_best.pth
fi
