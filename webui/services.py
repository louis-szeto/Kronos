"""Model loading, data loading, prediction logic, and chart generation."""

import os
import json
import datetime
import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils

from .config import DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# Model availability
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logger.warning("Kronos model cannot be imported, will use simulated data for demonstration")

# Mutable model state (module-level dict so routes can reassign)
_model_state = {
    'tokenizer': None,
    'model': None,
    'predictor': None,
}

# Convenience properties
def get_tokenizer():
    return _model_state['tokenizer']
def get_model():
    return _model_state['model']
def get_predictor():
    return _model_state['predictor']
def set_model_state(tokenizer, model, predictor):
    _model_state['tokenizer'] = tokenizer
    _model_state['model'] = model
    _model_state['predictor'] = predictor

# Available model configurations
# SEC-6: pinned HF revisions for supply-chain protection
_HF_PINNED_REVISIONS = {
    'NeoQuasar/Kronos-base': '2b55474',
    'NeoQuasar/Kronos-small': '901c26c',
    'NeoQuasar/Kronos-mini': 'f4e6869',
    'NeoQuasar/Kronos-Tokenizer-base': '0e01173',
    'NeoQuasar/Kronos-Tokenizer-2k': '26966d0',
}

def _get_pinned_revision(model_id):
    """Return pinned commit hash for a HuggingFace model, or None."""
    global_rev = os.environ.get('KRONOS_HF_REVISION')
    if global_rev:
        return global_rev
    return _HF_PINNED_REVISIONS.get(model_id)

AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file with path traversal protection."""
    try:
        resolved = Path(file_path).resolve()
        if not str(resolved).startswith(str(DATA_DIR)):
            logger.warning("Blocked path traversal attempt")
            return None, "Access denied"
        if not resolved.exists():
            return None, "File not found"
        if resolved.suffix == '.csv':
            df = pd.read_csv(resolved)
        elif resolved.suffix == '.feather':
            df = pd.read_feather(resolved)
        else:
            return None, "Unsupported file format"
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        df = df.dropna()
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        results_dir = str(RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        if actual_data and len(actual_data) > 0:
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]
                first_actual = actual_data[0]
                
                save_data['analysis']['continuity'] = {
                        'last_prediction': {
                            'open': last_pred['open'],
                            'high': last_pred['high'],
                            'low': last_pred['low'],
                            'close': last_pred['close']
                        },
                        'first_actual': {
                            'open': first_actual['open'],
                            'high': first_actual['high'],
                            'low': first_actual['low'],
                            'close': first_actual['close']
                        },
                        'gaps': {
                            'open_gap': abs(last_pred['open'] - first_actual['open']),
                            'high_gap': abs(last_pred['high'] - first_actual['high']),
                            'low_gap': abs(last_pred['low'] - first_actual['low']),
                            'close_gap': abs(last_pred['close'] - first_actual['close'])
                        },
                        'gap_percentages': {
                            'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                            'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                            'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                            'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                        }
                    }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Prediction results saved to: %s", filepath)
        return filepath
        
    except Exception as e:
        logger.error("Failed to save prediction results: %s", e)
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    if historical_start_idx + lookback + pred_len <= len(df):
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    if pred_df is not None and len(pred_df) > 0:
        if 'timestamps' in df.columns and len(historical_df) > 0:
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    if actual_df is not None and len(actual_df) > 0:
        if 'timestamps' in df.columns:
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    if 'timestamps' in historical_df.columns:
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
