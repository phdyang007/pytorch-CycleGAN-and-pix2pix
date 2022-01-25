###
 # @Author: Guojin Chen @ CUHK-CSE
 # @Homepage: https://dekura.github.io/
 # @Date: 2021-02-02 22:33:22
 # @LastEditTime: 2021-04-01 19:25:58
 # @Contact: cgjhaha@qq.com
 # @Description: transfer the data
###



python=/research/d2/xfyao/tools/anaconda3/envs/hdevel/bin/python

# $python src/data/target2ls.py \
# --in_dir /home/guojin/data/datasets/ganopc/data/artitgt \
# --out_dir /home/guojin/data/datasets/develset


$python src/data/target2ls.py \
--in_dir /research/d2/xfyao/guojin/data/datasets/ganopc_targets/targets \
--out_dir /research/d2/xfyao/guojin/data/datasets/develset_ganopc_train

