# TODO: create shell script for running your GAN/ACGAN model

wget -O model_GAN.pth https://www.dropbox.com/s/2f70ecemasxu3ox/GAN-G.pth?dl=1
wget -O model_ACGAN.pth https://www.dropbox.com/s/10p1mgms7vgajoh/ACGAN-G.pth?dl=1

python3 ./GAN_predict.py $1 model_GAN.pth
python3 ./ACGAN_predict.py $1 model_ACGAN.pth
