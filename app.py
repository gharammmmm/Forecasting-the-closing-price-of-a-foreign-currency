from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import joblib
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

app = Flask(__name__)

# تعريف نموذج NeuralNetwork
class NeuralNetwork(nn.Module):
    def __init__(self, num_feature):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(num_feature, 64, batch_first=True)
        self.fc = nn.Linear(64, num_feature)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden)
        return x

# تحميل النموذج
model = torch.load("saved_weights_all.pt", map_location=torch.device('cpu'), weights_only=False)
model.eval()  # وضع التقييم

# تحميل MinMaxScaler
scaler = joblib.load("scaler_all.pkl")
print("تم تحميل Scaler مع عدد الميزات:", scaler.n_features_in_)
print("✅ تم تحميل MinMaxScaler بنجاح!")
print("عدد الميزات التي تم تطبيق السكلر عليها:", scaler.n_features_in_)
print("القيم الدنيا المستخدمة في MinMaxScaler:", scaler.data_min_)
print("القيم العليا المستخدمة في MinMaxScaler:", scaler.data_max_)

# دالة معالجة البيانات
def preprocess_data(df):
    # التأكد من أن عمود "Date" موجود
    if "Date" not in df.columns:
        raise ValueError("الملف يجب أن يحتوي على عمود 'Date'.")

    # تحويل عمود التاريخ إلى datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

    df.drop(['Change %'], axis=1, inplace=True)

    # حذف أي صف يحتوي على تاريخ غير صالح
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = df['Date'].dt.date  # إزالة الوقت من التواريخ

    # عكس البيانات لضمان ترتيبها من الأقدم إلى الأحدث
    df = df.iloc[::-1]
    df.reset_index(inplace=True)
    df.rename(columns={'Vol': 'Volume'}, inplace=True)
    df.rename(columns={'Price': 'Close'}, inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    df['Volume'] = df['Volume'].astype('float64')

    # حساب المتوسطات المتحركة
    df["5d_sma"] = df["Close"].rolling(5).mean().fillna(df["Close"])
    df["9d_sma"] = df["Close"].rolling(9).mean().fillna(df["Close"])
    df["17d_sma"] = df["Close"].rolling(17).mean().fillna(df["Close"])

    return df

# دالة التنبؤ بالسعر
def predict_price(data):
    # تحويل البيانات إلى Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # التنبؤ
    with torch.no_grad():
        prediction = model(data_tensor)

    # تحويل النتيجة إلى numpy array
    predicted_price_scaled = prediction.numpy().flatten()

    # عكس تحجيم النتيجة للحصول على السعر النظامي
    predicted_price_original = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()

    return predicted_price_original

@app.route("/")
def home():
    return render_template("index.html")

# مسار التنبؤ
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # الحصول على الملف من الطلب
        file = request.files["file"]

        # قراءة البيانات
        df = pd.read_csv(file)

        # التحقق إذا كان الملف فارغًا أو لا يحتوي على البيانات المطلوبة
        if df is None or df.empty:
            return jsonify({"error": "الملف فارغ أو يحتوي على بيانات غير صالحة!"})

        # معالجة البيانات
        df = preprocess_data(df)

        # الحصول على التاريخ المدخل من المستخدم
        date_input = request.form["date"]
        date_input = datetime.strptime(date_input, "%Y-%m-%d").date()

        # التأكد من أن التاريخ موجود في البيانات
        if date_input not in df['Date'].values:
            return jsonify({"error": "التاريخ المدخل غير موجود في البيانات!"})

        # الحصول على الفهرس المناسب للتاريخ
        date_index = df[df['Date'] == date_input].index[0]

        # التأكد من أن هناك بيانات كافية قبل التاريخ المدخل
        if date_index < 11:
            return jsonify({"error": "لا يوجد بيانات كافية قبل التاريخ المدخل لإجراء التنبؤ!"})

        # استخراج الـ 11 يومًا السابقة بدلاً من 10
        data_input = df.iloc[date_index - 11: date_index][
            ["Open", "High", "Low", "5d_sma", "9d_sma", "17d_sma", "Close"]
        ].values

        # التحقق إذا كانت البيانات المدخلة غير فارغة
        if data_input is None or data_input.shape[0] == 0:
            return jsonify({"error": "حدث خطأ أثناء استخراج البيانات، تحقق من صحة الملف والتواريخ!"})

        # تحجيم البيانات بشكل منفصل لكل عمود
        data_input_scaled = np.zeros_like(data_input)  # إنشاء مصفوفة فارغة بنفس شكل البيانات
        for i in range(data_input.shape[1]):  # التكرار على كل عمود
            data_input_scaled[:, i] = scaler.transform(data_input[:, i].reshape(-1, 1)).flatten()

        # إجراء التنبؤ
        predicted_price = predict_price(data_input_scaled)

        # إرجاع السعر المتوقع للتاريخ المدخل
        return jsonify({"predicted_price": predicted_price[-1].tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)