# python -W ignore train-single.py --epochs 1000 --batch-size 4 --learning-rate 0.000001 --data_root './data/nuclei/BBBC008' --ckpt_path './checkpoints_single_1000/' --amp
python -W ignore train-multi.py --epochs 15000 --batch-size 2 --learning-rate 0.000001 --data_root './data/multi_refined_same_ch_wo23/' --ckpt_path './checkpoints_multi-5-3img_refined_15000_wo23_lastCAlastSA/' --amp