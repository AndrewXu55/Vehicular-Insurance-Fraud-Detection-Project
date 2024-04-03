from flask import Flask, request, render_template

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'jpg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    output = "Stuff"
    '''
    image = request.form['file']
    if not allowed_file(image.filename):
        output = "Invalid file format, plz use jpg"
    else:
        #do the analysis and prediction and file prep
        output = "it worked well brotha"
    '''
    return render_template('index.html', output=output) 


if __name__ == '__main__':
    app.run(debug=True)