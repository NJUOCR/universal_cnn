import os

from flask import Flask, request, make_response, jsonify

from processing.single_char_processing import Processor

app = Flask(__name__)
proc = Processor("/usr/local/src/data/stage2/all/all.json",
                 "/usr/local/src/data/stage2/all/aliasmap.json",
                 '/usr/local/src/data/stage2/all/ckpts',
                 64, 64, 4184, 64)


@app.route("/hello")
def hello():
    return 'hello ocr!'


@app.route("/")
def recognize_file():
    args = request.args
    path = args['path']
    if path[-4:] not in ('.jpg', '.png'):
        return make_response(jsonify({'error': 'only jpg, png files are supported'}, 415))
    if not os.path.isfile(path):
        return make_response(jsonify({'error': 'target file not found on server', 'target-file': path}, 404))

    txt = proc.process(path, p_thresh=0.9,
                       auxiliary_img='/usr/local/src/data/results/auxiliary.jpg',
                       auxiliary_html='/usr/local/src/data/results/auxiliary.html'
                       )
    return txt


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678, debug=False, threaded=False)
