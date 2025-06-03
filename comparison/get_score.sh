EPOCH=5
BATCH_SIZE=8
LR=0.001

########### UNet ###########
# 3subj
python train.py --model unet --epoch $EPOCH --batch_size $BATCH_SIZE --lr $LR --mode 3subj
python test_2d.py --model unet --batch_size $BATCH_SIZE --mode 3subj --weight unet.pth
python test_3d.py --model unet --batch_size $BATCH_SIZE --mode 3subj --weight unet.pth

# full
python train.py --model unet --epoch $EPOCH --batch_size $BATCH_SIZE --lr $LR --mode full
python test_2d.py --model unet --batch_size $BATCH_SIZE --mode full --weight unet.pth
python test_3d.py --model unet --batch_size $BATCH_SIZE --mode full --weight unet.pth

########### Swin ###########
# 3subj
python train.py --model swin --epoch $EPOCH --batch_size $BATCH_SIZE --lr $LR --mode 3subj
python test_2d.py --model swin --batch_size $BATCH_SIZE --mode 3subj --weight swin.pth
python test_3d.py --model swin --batch_size $BATCH_SIZE --mode 3subj --weight swin.pth

# full
python train.py --model swin --epoch $EPOCH --batch_size $BATCH_SIZE --lr $LR --mode full
python test_2d.py --model swin --batch_size $BATCH_SIZE --mode full --weight swin.pth
python test_3d.py --model swin --batch_size $BATCH_SIZE --mode full --weight swin.pth