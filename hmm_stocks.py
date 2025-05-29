import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime, timedelta

# Market state definitions
MARKET_STATES = {
    0: "Extreme Bear",
    1: "Strong Bear",
    2: "Moderate Bear",
    3: "Mild Bear",
    4: "Neutral",
    5: "Mild Bull",
    6: "Moderate Bull",
    7: "Strong Bull",
    8: "Extreme Bull",
    9: "Super Bull"
}

def get_state_description(state):
    """Get human-readable description of market state"""
    return MARKET_STATES[state]

def fetch_stock_data(ticker, start_date, end_date, interval='1d'):
    """Fetch stock data from Yahoo Finance with specified interval"""
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval=interval)
    return data

def prepare_features(data, timeframe='daily'):
    """Prepare features for HMM with adjustments for different timeframes"""
    if len(data) == 0:
        raise ValueError(f"No data available for {timeframe} timeframe")
    
    # Calculate returns
    returns = data['Close'].pct_change()
    
    # Adjust window sizes for different timeframes
    if timeframe == 'daily':
        vol_window = 20  # 20 days
        mom_window = 20
        rsi_window = 14
    elif timeframe == 'hourly':
        vol_window = 80  # 80 hours
        mom_window = 80
        rsi_window = 56
    else:  # minute
        vol_window = 240  # 240 minutes (4 hours)
        mom_window = 240
        rsi_window = 168  # 168 minutes (2.8 hours)
    
    # Calculate volatility
    volatility = returns.rolling(window=vol_window, min_periods=1).std()
    
    # Calculate momentum
    momentum = data['Close'].pct_change(periods=mom_window)
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Drop NaN values from all series
    returns = returns.dropna()
    volatility = volatility.dropna()
    momentum = momentum.dropna()
    rsi = rsi.dropna()
    
    # Align the indices to ensure same length
    common_index = returns.index.intersection(volatility.index).intersection(momentum.index).intersection(rsi.index)
    
    if len(common_index) == 0:
        raise ValueError(f"No common data points available for {timeframe} timeframe after feature calculation")
    
    returns = returns[common_index]
    volatility = volatility[common_index]
    momentum = momentum[common_index]
    rsi = rsi[common_index]
    
    # Combine features
    features = np.column_stack([returns, rsi]) # , volatility, momentum, rsi])
    
    return features

def fit_hmm(features, n_states=10):
    """Fit HMM to the data"""
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create and fit the HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(scaled_features)
    
    return model, scaler

def predict_states(model, features):
    """Predict hidden states"""
    return model.predict(features)

def calculate_accuracy(true_states, predicted_states):
    """Calculate prediction accuracy"""
    return np.mean(true_states == predicted_states)

def plot_results(data, states, returns, prediction_date=None):
    """Plot the results"""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Stock prices with market states
    plt.subplot(3, 1, 1)
    plt.plot(data.index[-len(states):], data['Close'][-len(states):], label='Stock Price')
    plt.title('Stock Price with Market States')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Create color map for states
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(MARKET_STATES)))
    
    # Add colored regions for market states
    for i in range(len(states)-1):
        state = states[i]
        plt.axvspan(data.index[-len(states)+i], data.index[-len(states)+i+1], 
                   alpha=0.3, color=colors[state], 
                   label=get_state_description(state) if i==0 else "")
    
    if prediction_date:
        plt.axvline(x=prediction_date, color='blue', linestyle='--', label='Prediction Date')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Returns distribution by state
    plt.subplot(3, 1, 2)
    for state in range(len(MARKET_STATES)):
        state_returns = returns[states == state]
        if len(state_returns) > 0:
            sns.kdeplot(data=state_returns, label=get_state_description(state), color=colors[state])
    plt.title('Returns Distribution by Market State')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: State probabilities
    plt.subplot(3, 1, 3)
    state_counts = pd.Series(states).value_counts().sort_index()
    plt.bar(range(len(MARKET_STATES)), state_counts, color=colors)
    plt.title('Distribution of Market States')
    plt.xlabel('Market State')
    plt.ylabel('Number of Days')
    plt.xticks(range(len(MARKET_STATES)), [get_state_description(s) for s in range(len(MARKET_STATES))], rotation=45)
    
    plt.tight_layout()
    plt.show()

def make_multi_timeframe_prediction(ticker, prediction_date):
    """Make predictions for daily, hourly, and minute data"""
    # Get training data
    train_start = (prediction_date - timedelta(days=500)).strftime('%Y-%m-%d')
    train_end = prediction_date.strftime('%Y-%m-%d')
    
    results = {}
    
    # Fetch and process data for each timeframe
    timeframes = {
        'daily': '1d',
        'hourly': '1h',
        'minute': '1m'
    }
    
    for timeframe, interval in timeframes.items():
        try:
            print(f"\nProcessing {timeframe} data...")
            data = fetch_stock_data(ticker, train_start, train_end, interval=interval)
            
            if len(data) == 0:
                print(f"Warning: No data available for {timeframe} timeframe")
                continue
                
            features = prepare_features(data, timeframe=timeframe)
            model, scaler = fit_hmm(features)
            state, probs = make_prediction(model, scaler, data, timeframe=timeframe)
            
            results[timeframe] = {
                'state': state,
                'probs': probs,
                'data': data,
                'model': model,
                'scaler': scaler,
                'features': features
            }
            
        except Exception as e:
            print(f"Error processing {timeframe} data: {str(e)}")
            continue
    
    if not results:
        raise ValueError("No timeframes could be processed successfully")
    
    return results

def make_prediction(model, scaler, data, timeframe='daily'):
    """Make prediction for the last period"""
    # Calculate features for the last period
    last_returns = data['Close'].pct_change().iloc[-1]
    
    # Adjust window size based on timeframe
    if timeframe == 'daily':
        window = 20
    elif timeframe == 'hourly':
        window = 80
    else:  # minute
        window = 2400
    
    last_volatility = data['Close'].pct_change().rolling(window=window).std().iloc[-1]
    last_momentum = data['Close'].pct_change(periods=window).iloc[-1]
    
    # Calculate RSI for the last period
    delta = data['Close'].diff()
    rsi_window = 14 if timeframe == 'daily' else (56 if timeframe == 'hourly' else 1680)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean().iloc[-1]
    rs = gain / loss
    last_rsi = 100 - (100 / (1 + rs))
    
    # Prepare features
    last_features = np.array([[last_returns, last_rsi]]) #, last_volatility, last_momentum, last_rsi]])
    last_features_scaled = scaler.transform(last_features)
    
    # Make prediction
    state = model.predict(last_features_scaled)[0]
    state_probs = model.predict_proba(last_features_scaled)[0]
    
    return state, state_probs

def plot_multi_timeframe_results(daily_data, hourly_data, minute_data, 
                               daily_states, hourly_states, minute_states,
                               daily_pred, hourly_pred, minute_pred, prediction_date):
    """Plot results for all three timeframes"""
    # Count available timeframes
    available_timeframes = sum(1 for x in [daily_data, hourly_data, minute_data] if x is not None)
    if available_timeframes == 0:
        print("No data available for plotting")
        return
        
    plt.figure(figsize=(20, 5 * available_timeframes))
    
    # Create color map for states
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(MARKET_STATES)))
    
    # Plot each available timeframe
    plot_idx = 1
    
    if daily_data is not None and daily_states is not None:
        plt.subplot(available_timeframes, 1, plot_idx)
        plt.plot(daily_data.index[-len(daily_states):], daily_data['Close'][-len(daily_states):], 
                 label='Daily Price')
        plt.title('Daily Market States')
        
        # Add colored regions for daily states
        for i in range(len(daily_states)-1):
            state = daily_states[i]
            plt.axvspan(daily_data.index[-len(daily_states)+i], daily_data.index[-len(daily_states)+i+1], 
                       alpha=0.3, color=colors[state],
                       label=get_state_description(state) if i==0 else "")
        
        if prediction_date:
            plt.axvline(x=prediction_date, color='blue', linestyle='--', label='Prediction Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_idx += 1
    
    if hourly_data is not None and hourly_states is not None:
        plt.subplot(available_timeframes, 1, plot_idx)
        plt.plot(hourly_data.index[-len(hourly_states):], hourly_data['Close'][-len(hourly_states):], 
                 label='Hourly Price')
        plt.title('Hourly Market States')
        
        # Add colored regions for hourly states
        for i in range(len(hourly_states)-1):
            state = hourly_states[i]
            plt.axvspan(hourly_data.index[-len(hourly_states)+i], hourly_data.index[-len(hourly_states)+i+1], 
                       alpha=0.3, color=colors[state],
                       label=get_state_description(state) if i==0 else "")
        
        if prediction_date:
            plt.axvline(x=prediction_date, color='blue', linestyle='--', label='Prediction Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plot_idx += 1
    
    if minute_data is not None and minute_states is not None:
        plt.subplot(available_timeframes, 1, plot_idx)
        plt.plot(minute_data.index[-len(minute_states):], minute_data['Close'][-len(minute_states):], 
                 label='Minute Price')
        plt.title('Minute Market States')
        
        # Add colored regions for minute states
        for i in range(len(minute_states)-1):
            state = minute_states[i]
            plt.axvspan(minute_data.index[-len(minute_states)+i], minute_data.index[-len(minute_states)+i+1], 
                       alpha=0.3, color=colors[state],
                       label=get_state_description(state) if i==0 else "")
        
        if prediction_date:
            plt.axvline(x=prediction_date, color='blue', linestyle='--', label='Prediction Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_hmm_structure(model, scaler, features):
    """Plot HMM structure including transition matrix, emission probabilities, and state characteristics"""
    plt.figure(figsize=(20, 15))
    
    # 1. Plot Transition Matrix
    plt.subplot(2, 2, 1)
    transition_matrix = model.transmat_
    sns.heatmap(transition_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdYlGn',
                xticklabels=[get_state_description(i) for i in range(len(MARKET_STATES))],
                yticklabels=[get_state_description(i) for i in range(len(MARKET_STATES))])
    plt.title('State Transition Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. Plot Emission Probabilities (Feature Distributions)
    plt.subplot(2, 2, 2)
    feature_names = ['Returns', 'Volatility', 'Momentum', 'RSI']
    emission_means = model.means_
    emission_covars = model.covars_
    
    # Plot means for each feature
    x = np.arange(len(feature_names))
    width = 0.8 / len(MARKET_STATES)
    
    for i in range(len(MARKET_STATES)):
        plt.bar(x + i*width, emission_means[i], width, 
                label=get_state_description(i),
                color=plt.cm.RdYlGn(i/len(MARKET_STATES)))
    
    plt.title('Emission Probabilities (Feature Means)')
    plt.xlabel('Features')
    plt.ylabel('Scaled Value')
    plt.xticks(x + width*len(MARKET_STATES)/2, feature_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Plot State Characteristics
    plt.subplot(2, 2, 3)
    state_means = np.mean(features, axis=0)
    state_stds = np.std(features, axis=0)
    
    # Plot feature distributions for each state
    for i in range(len(MARKET_STATES)):
        state_features = features[model.predict(scaler.transform(features)) == i]
        if len(state_features) > 0:
            plt.scatter(state_features[:, 0], state_features[:, 1], 
                       label=get_state_description(i),
                       color=plt.cm.RdYlGn(i/len(MARKET_STATES)),
                       alpha=0.5)
    
    plt.title('State Characteristics (Returns vs Volatility)')
    plt.xlabel('Returns')
    plt.ylabel('Volatility')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Plot State Persistence
    plt.subplot(2, 2, 4)
    state_sequence = model.predict(scaler.transform(features))
    state_durations = []
    current_state = state_sequence[0]
    duration = 1
    
    for state in state_sequence[1:]:
        if state == current_state:
            duration += 1
        else:
            state_durations.append((current_state, duration))
            current_state = state
            duration = 1
    state_durations.append((current_state, duration))
    
    # Calculate average duration for each state
    avg_durations = np.zeros(len(MARKET_STATES))
    state_counts = np.zeros(len(MARKET_STATES))
    
    for state, duration in state_durations:
        avg_durations[state] += duration
        state_counts[state] += 1
    
    avg_durations = np.divide(avg_durations, state_counts, 
                            out=np.zeros_like(avg_durations), 
                            where=state_counts!=0)
    
    plt.bar(range(len(MARKET_STATES)), avg_durations,
            color=[plt.cm.RdYlGn(i/len(MARKET_STATES)) for i in range(len(MARKET_STATES))])
    plt.title('Average State Duration (Days)')
    plt.xlabel('Market State')
    plt.ylabel('Average Duration (Days)')
    plt.xticks(range(len(MARKET_STATES)), 
               [get_state_description(s) for s in range(len(MARKET_STATES))],
               rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    ticker = "SPY"  # S&P 500 ETF
    prediction_date = datetime(2025, 2, 1)  # Date to predict
    
    try:
        # Make predictions for all timeframes
        results = make_multi_timeframe_prediction(ticker, prediction_date)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Date: {prediction_date.strftime('%Y-%m-%d')}")
        
        for timeframe, result in results.items():
            print(f"\n{timeframe.capitalize()} Prediction:")
            print(f"Predicted Market State: {get_state_description(result['state'])}")
            print("State Probabilities:")
            for i, prob in enumerate(result['probs']):
                print(f"{get_state_description(i)}: {prob:.2%}")
        
        # Plot results
        print("\nPlotting results...")
        plot_multi_timeframe_results(
            results.get('daily', {}).get('data'),
            results.get('hourly', {}).get('data'),
            results.get('minute', {}).get('data'),
            predict_states(results['daily']['model'], results['daily']['scaler'].transform(results['daily']['features'])) if 'daily' in results else None,
            predict_states(results['hourly']['model'], results['hourly']['scaler'].transform(results['hourly']['features'])) if 'hourly' in results else None,
            predict_states(results['minute']['model'], results['minute']['scaler'].transform(results['minute']['features'])) if 'minute' in results else None,
            results.get('daily', {}).get('state'),
            results.get('hourly', {}).get('state'),
            results.get('minute', {}).get('state'),
            prediction_date
        )
        
        # Plot HMM structure for all timeframes
        print("\nPlotting HMM structure...")
        for timeframe, result in results.items():
            print(f"\n{timeframe.capitalize()} HMM Structure:")
            plot_hmm_structure(result['model'], result['scaler'], result['features'])
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
