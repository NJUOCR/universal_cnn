import re
from typing import Generator


def rectify_by_location(char_gen: Generator):
    for char_list in char_gen:
        char = char_list[0]
        if char.c in ('，', '’'):
            char.set_content_text('，' if char.location() == 'floor' else '’',
                                  msg='[rectified] char is on %s' % char.location())


def rectify_3(char_gen: Generator):
    num_ptn = re.compile('\d')
    letter_ptn = re.compile('[a-zA-Z]')
    for char_3 in char_gen:
        left, char, right = char_3
        if char.c == right.c == '‘':
            char.set_content_text('“', msg='[rectified] `‘` on right')
            right.set_content_text('', msg='[rectified] merged into left')
        elif char.c == right.c == '’':
            char.set_content_text('”', msg='[rectified] `’` on right')
            right.set_content_text('', msg='[rectified] merged into left')
        elif char.c == 'O':
            num_score = bool(num_ptn.match(left.c)) + bool(num_ptn.match(right.c))
            letter_score = bool(letter_ptn.match(left.c)) + bool(letter_ptn.match(right.c))
            if num_score > letter_score:
                char.set_content_text('0', msg='[rectified] more numbers around %d>%d' % (num_score, letter_score))

        elif char.c == '0':
            num_score = bool(num_ptn.match(left.c)) + bool(num_ptn.match(right.c))
            letter_score = bool(letter_ptn.match(left.c)) + bool(letter_ptn.match(right.c))
            if letter_score > num_score:
                char.set_content_text('O', msg='[rectified] more letters around %d>%d' % (letter_score, num_score))
        elif char.c in ('I', 'l'):
            if right.c in ('、',):
                char.set_content_text('1', msg='[rectified] `、` on right')
