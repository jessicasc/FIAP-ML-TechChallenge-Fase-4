from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# iniciar api
app = FastAPI()

# carregar modelo e scaler
model = load_model("src/model.h5")

scaler = joblib.load("src/scaler.pkl")

# rota inicial para digitar o ticker
@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <html>

        <head>
            <title>Previsão de ações</title>
        </head>

        <body style="font-family: Arial; padding: 40px;">

            <h1>FIAP TC4 - Previsão de preço de ações com LSTM</h1>

            <form action="/predict" method="get">

                <input
                    type="text"
                    name="ticker"
                    placeholder="Digite o ticker"
                    required
                >

                <button type="submit">
                    Prever
                </button>

            </form>

        </body>

    </html>
    """

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