# Setup environment
1. `docker run --gpus all --shm-size 32G --name kaggle gcr.io/kaggle-gpu-images/python /bin/bash`
2. clone this repository & cd to it
3. `pip install -r requirements.txt`
4. `wandb login` if you want to track log
5. `mkdir ../input`

# Prepare data
1. `kaggle competitions download -c dfl-bundesliga-data-shootout`
2. `unzip -q dfl-bundesliga-data-shootout.zip -d input/dfl-bundesliga-data-shootout`
3. SoccerNetのデータをダウンロード
4. SoccerNetのデータの前処理
5. DFLのデータを前処理（画像に分割）

