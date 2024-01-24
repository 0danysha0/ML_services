from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Загрузка обученных моделей и векторизатора
model_category = joblib.load("model_label.joblib")
model_id = joblib.load("model_id.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Функция для предсказания категории товара по названию
def predict_category(title):
    # Векторизация текста
    title_vectorized = vectorizer.transform([title])
    # Предсказание категории
    category_prediction = model_category.predict(title_vectorized)
    return category_prediction[0]

# Функция для предсказания ID категории товара по названию
def predict_id(title):
    # Векторизация текста
    title_vectorized = vectorizer.transform([title])
    # Предсказание ID категории
    id_prediction = model_id.predict(title_vectorized)
    return id_prediction[0]

# Маршрут для отображения HTML-страницы
@app.route('/')
def home():
    return render_template('index.html')

# Основной маршрут для предсказания категории товара
@app.route('/predict_category', methods=['POST'])
def predict_category_route():
    data = request.get_json()
    title = data['title']
    prediction = predict_category(title)
    return jsonify({'category_prediction': prediction})

# Основной маршрут для предсказания ID категории товара
@app.route('/predict_id', methods=['POST'])
def predict_id_route():
    data = request.get_json()
    title = data['title']
    prediction_id = predict_id(title)
    return jsonify({'id_prediction': prediction_id})

if __name__ == '__main__':
    app.run(debug=True)

