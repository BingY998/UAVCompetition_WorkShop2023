LPN
python train.py \
--name='LPN+Pvt_v2+drn+two_branch-3' \
--data_dir='/media/user/1AD07E595CCB2146/zb/datasets/University-Release/train' \
--views=3 \
--droprate=0.75 \
--share \
--stride=1 \
--h=256 \
--w=256 \
--LPN \
--block=4 \
--lr=0.001 \
--gpu_ids='0'
#--fp16 \
#--extra \
#python test.py \
#--name='LPN+Pvt_v2+drn+two_branch-3' \
#--test_dir='/media/user/1AD07E595CCB2146/zb/datasets/University-Release/test' \
#--batchsize=16  \
#--gpu_ids='0'
#--test_dir='~/disk/zb/University-Release/test' \
# Baseline
# python train.py \
# --name='three_view_long_share_d0.75_256_s1_google_lr0.01' \
# --data_dir='/home/wangtyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.75 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --fp16 \
# --lr=0.01 \
# --gpu_ids='0'

# python test.py \
# --name='three_view_long_share_d0.75_256_s1_google_lr0.01' \
# --test_dir='/home/wangtyu/datasets/University-Release/test' \
# --gpu_ids='0'