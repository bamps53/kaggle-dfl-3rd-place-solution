# SoccerNet
python scripts/download_soccernet.py
unzip -q ../input/SoccerNet/tracking/train.zip -d ../input/SoccerNet/tracking/
unzip -q ../input/SoccerNet/tracking/test.zip -d ../input/SoccerNet/tracking/
rm ../input/SoccerNet/tracking/*.zip
mv ../input/SoccerNet/tracking/train/* ../input/SoccerNet/tracking/
mv ../input/SoccerNet/tracking/test/* ../input/SoccerNet/tracking/
python scripts/create_soccernet_df.py
python scripts/resize_soccernet.py

# DFL
kaggle competitions download -c dfl-bundesliga-data-shootout
unzip -q dfl-bundesliga-data-shootout.zip -d ../input/dfl-bundesliga-data-shootout
rm dfl-bundesliga-data-shootout.zip
python scripts/save_jpeg_images.py
python scripts/create_folds.py
python scripts/create_labels.py