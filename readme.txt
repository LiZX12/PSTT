Market1501
	1. data prepare
		python prepare --Market
	2. train
		python train_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "H:\program\Spatial-Temporal-Re-identification-master\dataset\market_rename"
	3. test
		python test_st_market.py --PCB --gpu_ids 0 --name ft_ResNet50_pcb_market_e --test_dir "H:\program\Spatial-Temporal-Re-identification-master\dataset\market_rename"

