from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch


def add_relation_network_config(cfg):
	# add configs here
	pass


def setup(args):
	cfg = get_cfg()
	add_relation_network_config(cfg)
	
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	
	default_setup(cfg, args)
	
	return cfg


def main(args);
	cfg = setup(args)
