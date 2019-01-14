#!/bin/bash
# Usage: ./test.sh <testing_options> <testing_data/feat> <output file>

python3 src/inference.py --caption_filename $1 --video_dir $2 \
--ensemble \
--models VideoCaption VideoCaption VideoCaption VCCommonSpace VCCommonSpace \
--args_filenames \
models/VideoCaption_margin0.01_all1024_cossim_valid/training_args.pkl \
models/VideoCaption_margin0.01_all1024_valid/training_args.pkl \
models/VideoCaption_margin0.01_mean_cossim_valid/training_args.pkl \
models/VCCommonSpace_cs512_margin0.01_all1024_valid/training_args.pkl \
models/VCCommonSpace_cs128_margin0.01_all1024_valid/training_args.pkl \
--model_filenames \
models/VideoCaption_margin0.01_all1024_cossim_valid/models_e517_vl0.0029.pt \
models/VideoCaption_margin0.01_all1024_valid/models_e669_vl0.0046.pt \
models/VideoCaption_margin0.01_mean_cossim_valid/models_e369_vl0.0029.pt \
models/VCCommonSpace_cs512_margin0.01_all1024_valid/models_e1510_vl0.0044.pt \
models/VCCommonSpace_cs128_margin0.01_all1024_valid/models_e959_vl0.0038.pt \
--output $3
