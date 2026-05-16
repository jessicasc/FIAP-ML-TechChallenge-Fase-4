from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import yfinance as yf
import numpy as np
import joblib
import time

# iniciar api
app = FastAPI()

# carregar modelo e scaler
model = load_model("src/model.h5")

scaler = joblib.load("src/scaler.pkl")

# rota inicial com frontend para entrada e saída dos dados
@app.get("/", response_class=HTMLResponse)
def home(ticker: str = None):

    prediction_text = ""

    if ticker:

        try:

            start_time = time.time()

            print(f"Nova requisição recebida, ticker: {ticker}")

            # baixar dados dos ultimos 120 dias corridos
            df = yf.download(ticker, period="120d")

            data = df[['Close']].copy()

            data.columns = ['Close'] 

            print(f"{len(data)} registros baixados da yfinance")

            # normalizacao
            scaled_data = scaler.transform(data)

            # ultimos 60 registros
            last_sequence = scaled_data[-60:]

            # formato (1, 60, 1)
            X = np.array([last_sequence])

            # previsao
            prediction = model.predict(X)

            # desnormalizar
            prediction = scaler.inverse_transform(prediction)

            predicted_prices = prediction[0]

            # log tempo de resposta
            end_time = time.time()

            response_time = end_time - start_time

            print(f"Previsões: {predicted_prices}")

            print(f"Tempo resposta: {response_time:.4f} segundos")

            # resultado HTML
            prediction_text = f"""

                <div
                    style="
                        margin-top: 20px;
                        padding: 20px;
                        border: 1px solid #ccc;
                        border-radius: 10px;
                        width: 400px;
                    "
                >

                    <p style="font-size: 14px; color: black;">
                        Previsão de preço de fechamento para os próximos 3 dias:
                    </p>

                    <ul>

                        <li>Dia 1: {predicted_prices[0]:.2f}</li>
                        <li>Dia 2: {predicted_prices[1]:.2f}</li>
                        <li>Dia 3: {predicted_prices[2]:.2f}</li>

                    </ul>

                </div>

            """

        except Exception as e:

            print(f"ERRO: {str(e)}")

            prediction_text = f"""

                <div
                    style="
                        margin-top: 20px;
                        padding: 20px;
                        border: 1px solid red;
                        border-radius: 10px;
                        width: 350px;
                    "
                >

                    <h3>
                        Erro ao processar previsão
                    </h3>

                    <p>{str(e)}</p>

                </div>

            """

    return f"""

    <html>

        <head>
            <title>Previsão de ações</title>
        </head>

        <body style="font-family: Arial; padding: 40px;">

            <h1>FIAP TC4 - Previsão de preço de ações com LSTM</h1>

            <form action="/" method="get">

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

            {prediction_text}

        </body>

    </html>

    """

# rota /predict para consumo de outras aplicações retorna json
@app.get("/predict")
def predict(ticker: str):

    start_time = time.time()

    print(f"Nova requisição recebida, ticker: {ticker}")

    # baixar dados dos ultimos 120 dias corridos
    df = yf.download(ticker,period="120d")

    data = df[['Close']].copy()

    data.columns = ['Close'] 

    print(f"{len(data)} registros baixados da yfinance")

    # normalizacao
    scaled_data = scaler.transform(data)

    # ultimos 60 registros
    last_sequence = scaled_data[-60:]

    # formato (1, 60, 1)
    X = np.array([last_sequence])

    # previsao
    prediction = model.predict(X)

    # desnormalizar
    prediction = scaler.inverse_transform(prediction)

    predicted_prices = prediction[0]

    # log tempo de resposta
    end_time = time.time()

    response_time = end_time - start_time

    print(f"Previsões: {predicted_prices}")

    print(f"Tempo resposta: {response_time:.4f} segundos")

    return {
        "ticker": ticker,
        "predicted_prices": [
            round(float(predicted_prices[0]), 2),
            round(float(predicted_prices[1]), 2),
            round(float(predicted_prices[2]), 2)    
        ],
        "response_time_seconds": round(response_time,4)
    }