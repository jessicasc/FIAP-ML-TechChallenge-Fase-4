from fastapi import FastAPI, HTTPException
import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# iniciar api
app = FastAPI()

# carregar modelo e scaler
model = load_model("src/model.h5")

scaler = joblib.load("src/scaler.pkl")

# rota predict recebe o ticker e retorna o preço futuro
@app.get("/predict")

def predict(ticker: str):

    try:

        # baixar dados dos ultimos 120 dias corridos
        df = yf.download(ticker, period="120d")

        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker não encontrado")

        data = df[['Close']]

        # normalizacao
        scaled_data = scaler.transform(data)

        # ultimos 60 registros
        last_sequence = scaled_data[-60:]

        # formato (1, 60, 1)
        X = np.array([last_sequence])

        # previsao
        prediction = model.predict(X)

        # desnormalizar
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # retorno
        return {
            "ticker": ticker,
            "predicted_price": round(float(predicted_price), 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )