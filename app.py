from flask import Flask, render_template, request, jsonify, redirect, url_for
from chat import get_response
import requests

app = Flask(__name__)

@app.get("/")
def main_page():
    return render_template('/index.html')

@app.post("/submit_form")
def submit_form():
    try:
        commands = request.form.get('commands')
        queries = request.form.getlist('queries')
        solutions = request.form.get('solutions')

        # Prepare data to send to connect.php
        data = {
            'commands': commands,
            'queries': queries,
            'solutions': solutions
        }

        # Send data to connect.php (update URL if needed)
        response = requests.post('http://localhost/PROJECT/connect1.php', data=data)  # Example port 80

        if response.status_code == 200:
            return redirect(url_for('main_page'))
        else:
            return f"Error submitting form: {response.text}", 500
    except requests.exceptions.RequestException as e:
        return f"Error processing form submission: {e}", 500
    except Exception as e:
        return f"Error processing form submission: {e}", 500

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    print(message)
    return jsonify(message)

app.run(debug=True,port=8000, host='0.0.0.0')