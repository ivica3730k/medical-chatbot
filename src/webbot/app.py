from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def response():
    msg = request.form["msg"]
    print(msg)
    return msg


if __name__ == "__main__":
    app.run()
