# FIAP ML Tech Challenge 4 - Previsão de preço de ações

O objetivo do projeto foi desenvolver um modelo de redes neurais LSTM (Long Short-Term Memory) que realiza previsão do preço de fechamento dos próximos 3 dias de uma ação com base em dados históricos (60 registros) obtidos através da biblioteca `yfinance`, disponibilizando o resultado por meio de uma API RESTful desenvolvida com FastAPI.

## Endpoints
**/** - Interface web para demonstração  
**/predict** - Rota que pode ser consumida por outras aplicações  

**Exemplo de uso:**    
/predict?ticker=AAPL  

**Exemplo de resposta:**  
{  
  "ticker": "AAPL",  
  "predicted_prices": [  
    281.21,  
    282.73,  
    284.10  
  ],   
  "response_time_seconds": 0.8421  
}  

O deploy da API foi feito usando Docker e Render, pode ser acessada por meio do link:  
https://fiap-ml-techchallenge-fase-4.onrender.com/  
  
Os detalhes de cada endpoint podem ser acessados na documentação Swagger disponível no link:    
https://fiap-ml-techchallenge-fase-4.onrender.com/docs  

## Tecnologias Utilizadas
- TensorFlow 
- yfinance
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- FastAPI
- Docker
- Render 

Link do vídeo com demonstração do projeto:  
  

`Jéssica Soares - RM 367045`
