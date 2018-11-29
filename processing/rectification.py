from typing import Generator, List

from utils.utext import TextChar
import re


def rectify_by_location(char_gen: Generator[List[TextChar]]):
    for char_list in char_gen:
        char = char_list[0]
        if char.c in ('，', '’'):
            char.c = '，' if char.location() == 'floor' else '’'


def rectify_3(char_gen: Generator[List[TextChar]]):
    num_ptn = re.compile('\d')
    letter_ptn = re.compile('[a-zA-Z]')
    for char_3 in char_gen:
        left, char, right = char_3
        if char == right == '‘':
            char.c = '“'
            right.valid(set_to=False)
        elif char == right == '’':
            char.c = '”'
            right.valid(set_to=False)
        elif char in ('O', '0'):
            num_score = bool(num_ptn.match(left.c)) + bool(num_ptn.match(right.c))
            letter_score = bool(letter_ptn.match(left.c)) + bool(letter_ptn.match(right.c))
            if num_score > letter_score:
                char.c = '0'
            if letter_score > num_score:
                char.c = 'O'
