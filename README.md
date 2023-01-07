## Detailed writeup
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236

## 1. Setup environment
```
docker run --gpus all --shm-size 32G --name kaggle gcr.io/kaggle-gpu-images/python /bin/bash
clone this repository 
cd kaggle-dfl-3rd-place-solution
pip install -r requirements.txt 
```

## 2. Prepare data
```
mkdir ../input
cd ../input
kaggle competitions download -c dfl-bundesliga-data-shootout
unzip -q dfl-bundesliga-data-shootout.zip -d input/dfl-bundesliga-data-shootout
cd ../kaggle-dfl-3rd-place-solution
bash ./prepare_data.sh
```

## 3. Pretrain on SoccerNet
```
python pretrain.py -c configs.pretrain_b2
python pretrain.py -c configs.pretrain_b3_dwc
python pretrain.py -c configs.pretrain_b4
```

## 4. Main
```
python final_all_mix_360_b2_dur64_noreg_nomask.py
python stage0_210_b2_d32_mixup_reg_lr_tune_from_ball022_ema_360_640_fixed_deep.py --start_fold 0 --end_fold 1
python stage0_217_b2_d80_ft210.py --start_fold 0 --end_fold 1
python stage0_229_b3_d32_mixup_360_dwc_no_mask_no_reg.py --start_fold 0 --end_fold 1
python stage0_229_b3_d32_mixup_360_dwc_no_mask_no_reg_ft128.py --start_fold 0 --end_fold 1
python final_fold2_mix_320_b4_dur32_nohm_deep.py --start_fold 2 --end_fold 3
python final_fold2_mix_320_b4_dur32_nohm_deep_ft128.py --start_fold 2 --end_fold 3
python final_fold3_mix_320_b2_dur64_nohm_deep_lstm.py --start_fold 3 --end_fold 4
python final_fold3_mix_320_b2_dur64_nohm_deep_lstm_ft128.py --start_fold 3 --end_fold 4
python final_fold4_mix_320_b2_dur64_nohm_deep_lstm.py --start_fold 4 --end_fold 5
python final_fold5_mix_480_b2_dur32_deep_lstm_noreg.py --start_fold 5 --end_fold 6
```

## 5. Datasets (models used in best submission)
https://www.kaggle.com/datasets/bamps53/stage0217  
https://www.kaggle.com/datasets/bamps53/stage0-229  
https://www.kaggle.com/datasets/bamps53/final-fold2-mix-320-b4-dur32-ft128  
https://www.kaggle.com/datasets/bamps53/final-fold3-mix-320-b2-dur64-lstm-ft128  
https://www.kaggle.com/datasets/bamps53/final-fold5-mix-480-b2-dur32-lstm-noreg  
https://www.kaggle.com/datasets/bamps53/final-all  

## 6. Submit
https://www.kaggle.com/code/bamps53/final-sub?scriptVersionId=107978496
