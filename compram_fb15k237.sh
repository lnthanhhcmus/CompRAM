python run.py -epoch 1500 -name compram_fb15k237 -model compram -score_func \
complex -opn sub -gpu 0 -data FB15k-237 -gcn_drop 0.4 -batch 1024 \
-attention True -head_num 1 -init_dim 100 -gcn_dim 200 -embed_dim 200 \
-lr 0.001 -gcn_layer 2 -g 12.0 -init_e n -init_r u