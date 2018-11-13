import argparse
import yaml
#
#
# parse = argparse.ArgumentParser()
# parse.add_argument("--mode", default='train', help='train | infer')
# #
# cmd_args = parse.parse_args()

f = open('configs/single_char.yaml', encoding='utf-8')
# f = open('configs/punctuation_letter_digit.yaml', encoding='utf-8')
args = yaml.load(f.read())
f.close()
