# ============================================================
#  SPAM DETECTOR - Flask Web App
#  Cách chạy: python app.py
#  Mở trình duyệt: http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, request, jsonify
import pickle, re, os, traceback

# --- NLP (phải trùng với preprocessing lúc train) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# ── Đường dẫn tới các file model ──────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'spam_model.pkl')
TFIDF_PATH  = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# ── Load model khi khởi động server ───────────────────────
model, tfidf, le = None, None, None

def load_models():
    global model, tfidf, le
    try:
        with open(MODEL_PATH,   'rb') as f: model = pickle.load(f)
        with open(TFIDF_PATH,   'rb') as f: tfidf = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f: le    = pickle.load(f)
        print("✅ Tải model thành công!")
    except FileNotFoundError as e:
        print(f"⚠️  Chưa tìm thấy model: {e}")
        print("   → Hãy đặt 3 file .pkl vào thư mục /models/")

load_models()

# ── Hàm tiền xử lý (PHẢI GIỐNG với lúc train) ─────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return ' '.join(tokens)

# ── Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Thiếu nội dung tin nhắn'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'Tin nhắn không được để trống'}), 400
    if len(message) > 5000:
        return jsonify({'error': 'Tin nhắn quá dài (tối đa 5000 ký tự)'}), 400

    if model is None:
        return jsonify({'error': 'Model chưa được tải. Kiểm tra thư mục /models/'}), 503

    try:
        clean   = preprocess(message)
        vec     = tfidf.transform([clean])
        pred    = model.predict(vec)[0]
        label   = le.inverse_transform([pred])[0].upper()

        # Tính confidence
        confidence = None
        try:
            proba      = model.predict_proba(vec)[0]
            confidence = round(float(max(proba)) * 100, 1)
        except AttributeError:
            try:
                score      = model.decision_function(vec)[0]
                confidence = round(min(99.9, abs(float(score)) * 20 + 70), 1)
            except Exception:
                confidence = None

        return jsonify({
            'label':      label,
            'is_spam':    label == 'SPAM',
            'confidence': confidence,
            'clean_text': clean,
            'char_count': len(message),
            'word_count': len(message.split()),
        })

    except Exception:
        return jsonify({'error': f'Lỗi xử lý: {traceback.format_exc()}'}), 500

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({'loaded': False})
    name = type(model).__name__
    return jsonify({'loaded': True, 'model_type': name})

if __name__ == '__main__':
    print("🚀 Khởi động Spam Detector...")
    print("   Mở trình duyệt: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
