import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# configuracoes
tickers = [
    "AAPL",
    "PETR4.SA",
    "MGLU3.SA"
]

sequence_length = 60

future_days = 3

# baixar dados da biblioteca yfinance para treinamento

all_data = []

ticker_data = {}

for ticker in tickers:

    print(f"Baixando dados: {ticker}")

    df = yf.download(
        ticker,
        start="2021-01-01",
        end="2025-12-31"
    )

    
    data = df[['Close']].copy()

    data.columns = ['Close'] 

    ticker_data[ticker] = data

    all_data.append(data)

final_data = pd.concat(all_data, ignore_index=True)

# normalizacao 
scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(final_data)

# criar sequencias temporais com 60 dias cada 
X = []
y = []

for ticker in tickers:

    data = ticker_data[ticker]

    scaled_data = scaler.transform(data)

    for i in range(sequence_length, len(scaled_data) - future_days):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i:i+future_days, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# dividir treino e teste
train_size = int(len(X) * 0.7)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# criar modelo LSTM
model = Sequential()

model.add(LSTM(units=64, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=64))

model.add(Dropout(0.2))

model.add(Dense(future_days))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# treinar modelo
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# predicoes
predictions = model.predict(X_test)

# desfazer normalização
predictions = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test)

# metricas
mae = mean_absolute_error(real_prices, predictions)

rmse = np.sqrt(mean_squared_error(real_prices, predictions))

mape = mean_absolute_percentage_error(real_prices, predictions)

print(f"MAE: {mae:.2f}")

print(f"RMSE: {rmse:.2f}")

print(f"MAPE: {mape * 100:.2f}%")

plt.figure(figsize=(14, 6))
plt.plot(real_prices[:, 0],label='Real Dia 1')
plt.plot(predictions[:, 0],label='Previsto Dia 1')
plt.title(f'{ticker} - Previsão Próximos 3 Dias')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()

# salvar modelo
model.save("model.h5")

# salvar scaler
joblib.dump(scaler, "scaler.pkl")

print("Modelo salvo com sucesso!")