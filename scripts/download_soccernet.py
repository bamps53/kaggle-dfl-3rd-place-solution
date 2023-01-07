from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="../input/SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test"])