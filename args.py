import yaml


f = open('configs/single_char.yaml', encoding='utf-8')
# f = open('configs/punctuation_letter_digit.yaml', encoding='utf-8')
args = yaml.load(f.read())
f.close()
