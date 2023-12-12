import os
import joblib
from scanner import main
from flask import Flask, render_template, request
from vector import vectorizer
app = Flask(__name__, template_folder="templates")

@app.route("/")
def hello():
    return render_template("base.html")

@app.route("/", methods=['POST','GET'])
def result():
    comment=request.form['comment']
    mode_name = "comment_scanner_model"

    if os.path.exists(mode_name):           #if the model is already created, we will load it otherwise new model
        dtree = joblib.load(mode_name)
        comment_vect=vectorizer(comment)
        prediction=dtree.predict(comment_vect[0])
        prediction=str(prediction)
    else:
        prediction=main(comment)
        prediction=str(prediction)
    prediction=prediction.replace("[","").replace("]","").replace("'","")
    return render_template("result.html",prediction=prediction,comment=comment)

if __name__=="__main__":
    app.run(debug=True)
