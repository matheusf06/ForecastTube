import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def load_data(dates):
    """Carrega as datas fornecidas em um DataFrame."""
    month_mapping = {
        "jan.": "Jan",
        "fev.": "Feb",
        "mar.": "Mar",
        "abr.": "Apr",
        "mai.": "May",
        "jun.": "Jun",
        "jul.": "Jul",
        "ago.": "Aug",
        "set.": "Sep",
        "out.": "Oct",
        "nov.": "Nov",
        "dez.": "Dec"
    }
    for month, abbr in month_mapping.items():
        dates = [date.replace(month, abbr) for date in dates]
    dates = [datetime.strptime(date, "%d de %b de %Y") for date in dates]
    df = pd.DataFrame(dates, columns=['data'])
    df.set_index('data', inplace=True)
    return df

def evaluate_model(weekly_data):
    """Avalia o modelo usando Time Series Cross Validation."""
    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    for train_index, test_index in tscv.split(weekly_data):
        train_data, test_data = weekly_data.iloc[train_index], weekly_data.iloc[test_index]
        model = ARIMA(train_data, order=(1, 1, 1))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_data))
        error = mean_squared_error(test_data, predictions)
        errors.append(error)
    return errors

def generate_predictions(model_fit):
    """Gera previsões para as próximas semanas."""
    today = datetime.now()
    next_week = model_fit.forecast(steps=4)
    predicted_dates = [today + timedelta(weeks=i) for i in range(1, 5)]
    return predicted_dates

def main():
    # Datas fornecidas
    dates = [
        "22 de mar. de 2024",
        "14 de mar. de 2024",
        "6 de mar. de 2024",
        "1 de mar. de 2024",
        "27 de fev. de 2024",
        "21 de fev. de 2024",
        "16 de fev. de 2024",
        "12 de fev. de 2024",
        "1 de dez. de 2023",
        "28 de fev. de 2023",
        "7 de fev. de 2023",
        "3 de fev. de 2023",
        "30 de jan. de 2023",
        "11 de jan. de 2023",
        "8 de jan. de 2023",
        "30 de dez. de 2022",
        "24 de dez. de 2022",
        "13 de dez. de 2022",
        "4 de dez. de 2022",
        "1 de dez. de 2022",
        "20 de nov. de 2022",
        "15 de nov. de 2022",
        "10 de nov. de 2022",
        "1 de nov. de 2022"
    ]
    df = load_data(dates)
    weekly_data = df.resample('W').size()

    model = ARIMA(weekly_data, order=(1,1,1))
    model_fit = model.fit()

    errors = evaluate_model(weekly_data)
    print("Mean Squared Errors:", errors)

    predicted_dates = generate_predictions(model_fit)

    print("\nPróximas datas de postagem previstas:")
    for date in predicted_dates:
        print(date.strftime('%d de %b de %Y'))

if __name__ == "__main__":
    main()
