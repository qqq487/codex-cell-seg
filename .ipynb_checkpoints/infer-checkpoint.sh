# python predict-single.py --model 'checkpoints_single_1000/checkpoint_epoch901.pth' --input_folder 'test_imgs_single_nuclei/' --mask-threshold 0.5 --scale 1
python predict-multi.py --model 'checkpoints_multi-5-3img_refined_15000_wo23_lastCAlastSA/checkpoint_epoch4001.pth' --input_folder 'test_imgs_multi/' --mask-threshold 0.5 --scale 1
