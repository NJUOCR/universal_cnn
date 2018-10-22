# import argparse
#
#
# parse = argparse.ArgumentParser()
#
# parse.add_argument("--gpu", type=int, default=-1, help='which gpu for running. On cpu if -1')
# parse.add_argument("--mode", default='train', help='train | infer')
#
# args = parse.parse_known_args()
import json

import yaml
f = open('config-test.yaml', encoding='utf-8')
args = yaml.load(f.read())
f.close()
