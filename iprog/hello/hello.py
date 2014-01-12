from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route('/hello/postform')
def getPostForm():
    return render_template('form.html')


@app.route('/form', methods=['POST','GET'])
def processForm():
    if request.method == 'POST':
        return 'Welcome ' + request.form['firstname'] + ' ' + request.form['lastname'] + '!'
    else:
        return render_template('form.html')


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/')
def hello_world():
    return 'Hello World!'



if __name__ == '__main__':
    app.run(debug=True)
