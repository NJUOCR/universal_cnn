import argparse
import json

from data import SingleCharData
from main import Main

parse = argparse.ArgumentParser()

parse.add_argument("--src_dir", type=str, default="/usr/local/src/data/sc_pld/val",
                   help='the directory containing all the char images to be tested')

parse.add_argument("--ckpt_dir", type=str, default="./ckpts/single_pld_3990")

parse.add_argument("--out", type=str, default="./accuracy.json")
args = parse.parse_args()

if __name__ == '__main__':

    data = SingleCharData(64, 64)
    data.load_char_map("label_maps/single_pld_3990.json").read(args.src_dir).init_indices()
    main = Main()
    results = main.infer(data, batch_size=64, ckpt_dir=args.ckpt_dir)
    labels = data.unmap(data.labels)

    char_map = {}
    for result, label in zip(results, labels):
        # for result, label, img in zip(results, labels, data.images):
        #     show(img.reshape(img.shape[:2]))
        result = result[0]
        if label not in char_map:
            char_map[label] = {
                'total': 0,
                'bingo': 0
            }
        char_map[label]['total'] += 1
        char_map[label]['bingo'] += (1 if result == label else 0)
        print(result, label)

    acc = {}
    for char, obj in char_map.items():
        acc[char] = obj['bingo'] / obj['total']
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)
