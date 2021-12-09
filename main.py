from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("self_coco_train", {}, "/home/alex/Pictures/coco/annotations/instances_train2017.json", "/home/alex/Pictures/coco/train2017")
register_coco_instances("self_coco_val", {}, "/home/alex/Pictures/coco/annotations/instances_val2017.json", "/home/alex/Pictures/coco/val2017")

coco_val_metadata = MetadataCatalog.get("self_coco_val")
dataset_dicts = DatasetCatalog.get("self_coco_val")
print(coco_val_metadata)
