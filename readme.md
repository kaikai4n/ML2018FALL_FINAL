# How to train
- example
    ```
    python3 src/train.py --prefix=VCCommonSpace_margin0.01_valid --model=VCCommonSpace --margin=0.01 --save_intervals=100 --epoches=300 --validation --video_dir=data/training_data/feat --caption_filename=data/training_label.json
    ```
- When training for the first time, please note to add ``--no_load_word_dict`` arguments to create word dictionary.
# How to test
- example
    ```
    python3 src/inference.py --caption_filename=data/testing_options.csv --video_dir=data/testing_data/feat/ --model=VCCommonSpace --args_filename=models/VCCommonSpace_margin0.01_valid/training_args.pkl --model_filename=models/VCCommonSpace_margin0.01_valid/models_e300.pt --output=ans_VCCommonSpace_margin0.01_e300_valie.csv
    ```
