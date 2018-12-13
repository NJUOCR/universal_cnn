import json
import os

import yaml
from flask import Flask, request, make_response, jsonify, render_template, Response, send_file

from processing.single_char_processing import Processor

app = Flask(__name__)
with open('configs/infer.yaml', encoding='utf-8') as conf_file:
    conf_args = yaml.load(conf_file)
CONF = {**{
    'charmap_path': "/usr/local/src/data/stage2/all/all.json",
    'aliasmap_path': "/usr/local/src/data/stage2/all/aliasmap.json",
    'ckpt_dir': "/usr/local/src/data/stage2/all/ckpts",
    'input_height': 64,
    'input_width': 64,
    'num_class': 4184,
    'batch_size': 64,
    'p_thresh': 0.8
}, **conf_args}
print(json.dumps(CONF, ensure_ascii=False, indent=2))
proc = Processor(CONF['charmap_path'], CONF['aliasmap_path'], CONF['ckpt_dir'],
                 CONF['input_height'], CONF['input_width'], CONF['num_class'], CONF['batch_size'])


@app.route("/hello")
def hello():
    return 'hello ocr!'


@app.route("/")
def recognize_file():
    args = request.args
    path = args['path']
    auxiliary = 'auxiliary' in args and bool(args['auxiliary'])
    with_log = 'logs' in args and bool(args['logs'])
    if path[-4:] not in ('.jpg', '.png'):
        return make_response(jsonify({'error': 'only jpg, png files are supported'}, 415))
    if not os.path.isfile(path):
        return make_response(jsonify({'error': 'target file not found on server', 'target-file': path}, 404))

    if not with_log:
        rs = proc.get_text_result(path, p_thresh=CONF['p_thresh'],
                                  auxiliary_img='./static/auxiliary.jpg' if auxiliary else None)
    else:
        rs = proc.get_json_result(path, p_thresh=CONF['p_thresh'],
                                  auxiliary_img='./static/auxiliary.jpg' if auxiliary else None)
    return rs if not auxiliary else Response(
        json.dumps({
            'rs': rs,
            'img': '/static/auxiliary.jpg'
        }, ensure_ascii=False),
        mimetype='application/json'
    )


@app.route("/debugger")
def debugger():
    return send_file('templates/debugger.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=False, threaded=False)
