python run.py -epoch 1500 -name compram_wn18rr -model compram -score_func \
complex -opn sub -gpu 0 -data WN18RR -gcn_drop 0.4 -batch 256 \
-attention True -head_num 1 -init_dim 100 -gcn_dim 200 -embed_dim 200 \
-lr 0.001 -gcn_layer 1 -g 6.0 -init_e u -init_r n