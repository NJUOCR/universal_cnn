import re
from typing import Generator


def rectify_by_location(char_gen: Generator):
    for char_list in char_gen:
        char = char_list[0]
        if char.c in ('，', '’'):
            char.set_content_text('，' if char.location() == 'floor' else '’',
                                  msg='char is on %s' % char.location())
        if char.c in ('o', '。'):
            char.set_content_text('。' if char.location() == 'floor' else 'o',
                                  msg='char is on %s' % char.location())


def rectify_5(char_gen: Generator):
    """
    `‘`+`‘` = `“`
    `’`+`’` = `”`
    :param char_gen:
    :return:
    """
    num_ptn = re.compile('\d')
    letter_ptn = re.compile('[a-zA-Z]')
    for char_5 in char_gen:
        lefter, left, char, right, righter = char_5
        if char.c == right.c == '‘':
            char.set_content_text('“', msg='`‘` on right')
            right.set_content_text('', msg='merged into left')
        elif char.c == right.c == '’':
            char.set_content_text('”', msg='`’` on right')
            right.set_content_text('', msg='merged into left')
        elif char.c in ('O', '。'):
            num_score = sum(map(lambda _: _.c is not None and bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
            letter_score = sum(
                map(lambda _: _.c is not None and bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
            if num_score > letter_score:
                char.set_content_text('0', msg='more numbers around %d>%d' % (num_score, letter_score))
        elif char.c == '0':
            num_score = sum(map(lambda _: _.c is not None and bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
            letter_score = sum(
                map(lambda _: _.c is not None and bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
            if letter_score > num_score:
                char.set_content_text('O', msg='more letters around %d>%d' % (letter_score, num_score))
        if char.c in ('I', 'l', ']'):
            if right.c in ('、',):
                char.set_content_text('1', msg='`、` on right')
            else:
                num_score = sum(
                    map(lambda _: _.c is not None and bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
                letter_score = sum(
                    map(lambda _: _.c is not None and bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
                if num_score > letter_score:
                    char.set_content_text('1', msg='more numbers around %d>%d' % (num_score, letter_score))
        if char.c in ('S', 's'):
            num_score = sum(
                map(lambda _: _.c is not None and bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
            letter_score = sum(
                map(lambda _: _.c is not None and bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
            if num_score > letter_score:
                char.set_content_text('5', msg='more numbers around %d>%d' % (num_score, letter_score))

        if char.c in ('d', '4') and right.c == '、':
            char.set_content_text('小', msg='merge "d|4、" to "小"')
            right.set_content_text('', msg='merge "d|4、" to "小"')
        if char.c in ('i',) and right.c == '、':
            char.set_content_text('门', msg='merge "i、" to "门"')
            right.set_content_text('', msg='merge "i、" to "门"')

        char_5_c = set(map(lambda x: x.c, char_5))
        if len(char_5_c.intersection(('年', '月', '日', '时', '分', '秒'))) > 0:
            for char in char_5:
                if char.c in ('I', 'l', '!', '！'):
                    char.set_content_text('1')
                if char.c in ('O', ):
                    char.set_content_text('0')


