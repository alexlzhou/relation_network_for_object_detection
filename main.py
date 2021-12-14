from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

import cv2 as cv
import os
import random


def train_val(train, val):
	register_coco_instances("self_coco_train", {}, "/home/alex/Pictures/coco/annotations/instances_train2017.json", "/home/alex/Pictures/coco/train2017")
	register_coco_instances("self_coco_val", {}, "/home/alex/Pictures/coco/annotations/instances_val2017.json", "/home/alex/Pictures/coco/val2017")

	coco_val_metadata = MetadataCatalog.get("self_coco_val")
	dataset_dicts = DatasetCatalog.get("self_coco_val")

	'''
	for d in random.sample(dataset_dicts, 1):
		img = cv.imread(d["file_name"])
		visualizer = Visualizer(img[:, :, ::-1], metadata=coco_val_metadata, scale=0.5)
		vis = visualizer.draw_dataset_dict(d)
		cv.imshow("image", vis.get_image()[:, :, ::-1])
		cv.waitKey()
	'''
	
	cfg = get_cfg()

	cfg.DATASETS.TRAIN = ("self_coco_train", )
	cfg.DATASETS.TEST = ("self_coco_val", )

	cfg.DATALOADER.NUM_WORKERS = 1

	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.0025
	cfg.SOLVER.MAX_ITER = 20000

	cfg.TEST.EVAL_PERIOD = 100

	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=False)
	
	if train:
		trainer.train()

	print(coco_val_metadata)
	
	if val:
		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
		predictor = DefaultPredictor(cfg)

		for d in random.sample(dataset_dicts, 1):
			img = cv.imread(d["file_name"])
			outputs = predictor(img)
			v = Visualizer(img, metadata=coco_val_metadata, scale=1.2, instance_mode=ColorMode.IMAGE)
			print(outputs["instances"].pred_classes)
			print(outputs["instances"].pred_boxes)
			
			v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
			cv.imshow("image", v.get_image())
			cv.waitKey()


train_val(False, True)
