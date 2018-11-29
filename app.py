import os
import json
import yaml
from flask import Flask, request, make_response, jsonify, render_template

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
# proc = Processor("/usr/local/src/data/stage2/all/all.json",
#                  "/usr/local/src/data/stage2/all/aliasmap.json",
#                  '/usr/local/src/data/stage2/all/ckpts',
#                  64, 64, 4184, 64)
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
    auxiliary = bool(args['auxiliary'])
    if path[-4:] not in ('.jpg', '.png'):
        return make_response(jsonify({'error': 'only jpg, png files are supported'}, 415))
    if not os.path.isfile(path):
        return make_response(jsonify({'error': 'target file not found on server', 'target-file': path}, 404))

    txt = proc.process(path, p_thresh=CONF['p_thresh'],
                       auxiliary_img='./static/auxiliary.jpg' if auxiliary else None,
                       auxiliary_html='./static/auxiliary.html' if auxiliary else None
                       )
    return txt if not auxiliary else jsonify(txt=txt, img='/static/auxiliary.jpg', html='/static/auxiliary.html')


@app.route("/debugger")
def debugger():
    return render_template('debugger.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=False, threaded=False)
