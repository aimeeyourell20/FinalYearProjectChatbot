from flask import Flask, request, jsonify

from main import chatWithBot

app = Flask(__name__)

#Allows us to connect to Android Studio
@app.route('/chat', methods=['GET', 'POST'])
def chatBot():
    chatInput = request.form['input']
    return jsonify(reply=chatWithBot(chatInput))


if __name__ == '__main__':
    app.run(host='172.20.10.3', debug=True)