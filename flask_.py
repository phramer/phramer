# predictor_app.py
import flask
from flask import request
#from predictor_api import make_prediction

# Initialize the app

app = flask.Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello():
    print(request.args)
    if (request.args):
        print('input_value:', request.args["value"])
        return "got input value"
    else:
        print('no input_value')
        return "no value"

# @app.route("/", methods=["GET","POST"])
# def predict():
#    print(request.args)
#    if(request.args):
#        x_input, predictions = make_prediction(request.args['chat_in'])
#        print(x_input)
#        return flask.render_template('predictor.html',
#                                     chat_in=x_input,
#                                     prediction=predictions)
#    else:
#        x_input, predictions = make_prediction('')
#        return flask.render_template('predictor.html',
#                                     chat_in=x_input,
#                                     prediction=predictions)

if __name__=="__main__":
        app.run(host='0.0.0.0', port=8383)

