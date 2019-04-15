import yaml


f = open('configs/single_char_3991.yaml', encoding='utf-8')
# f = open('configs/punctuation_letter_digit.yaml', encoding='utf-8')
args = yaml.load(f.read())
f.close()
