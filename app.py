from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load Model once when the server starts
model_name = "TusharJoshi89/title-generator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Quantize for CPU speed
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

@app.route('/')
def index():
    # Renders your landing page
    return render_template('index.html')

@app.route('/generator')
def generator_page():
    # Renders the actual generation tool page
    return render_template('generator.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    abstract = data.get('abstract', '')
    
    if not abstract.strip():
        return jsonify({"error": "Abstract is empty"}), 400

    input_text = "summarize: " + abstract
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs, 
        max_length=40, 
        num_beams=5, 
        num_return_sequences=3, 
        early_stopping=True
    )

    titles = [tokenizer.decode(out, skip_special_tokens=True).title() for out in outputs]
    return jsonify({"titles": titles})

if __name__ == '__main__':
    # Setting use_reloader=False prevents the 'SystemExit' error in Jupyter
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)