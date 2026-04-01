"""Flask route definitions for Kronos WebUI."""

import os
import secrets
import logging

import pandas as pd
from flask import Flask, render_template, request, jsonify
from functools import wraps

from .config import API_KEY
from .services import (
    MODEL_AVAILABLE, AVAILABLE_MODELS,
    KronosTokenizer, Kronos, KronosPredictor,
    _get_pinned_revision,
    load_data_files, load_data_file,
    save_prediction_results, create_prediction_chart,
    get_predictor, set_model_state,
)

logger = logging.getLogger(__name__)

def require_api_key(f):
    """Decorator: require X-API-Key header for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key')
        if not key or not secrets.compare_digest(key, API_KEY):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def register_routes(app):
    """Register all routes on the Flask app."""

    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')

    @app.route('/api/data-files')
    def get_data_files():
        """Get available data file list"""
        data_files = load_data_files()
        return jsonify(data_files)

    @app.route('/api/load-data', methods=['POST'])
    @require_api_key
    def load_data():
        """Load data file"""
        try:
            data = request.get_json()
            file_path = data.get('file_path')
            
            if not file_path:
                return jsonify({'error': 'File path cannot be empty'}), 400
            
            df, error = load_data_file(file_path)
            if error:
                return jsonify({'error': error}), 400
            
            def detect_timeframe(df):
                if len(df) < 2:
                    return "Unknown"
                
                time_diffs = []
                for i in range(1, min(10, len(df))):
                    diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                    time_diffs.append(diff)
                
                if not time_diffs:
                    return "Unknown"
                
                avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
                
                if avg_diff < pd.Timedelta(minutes=1):
                    return f"{avg_diff.total_seconds():.0f} seconds"
                elif avg_diff < pd.Timedelta(hours=1):
                    return f"{avg_diff.total_seconds() / 60:.0f} minutes"
                elif avg_diff < pd.Timedelta(days=1):
                    return f"{avg_diff.total_seconds() / 3600:.0f} hours"
                else:
                    return f"{avg_diff.days} days"
            
            data_info = {
                'rows': len(df),
                'columns': list(df.columns),
                'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
                'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
                'price_range': {
                    'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                    'max': float(df[['open', 'high', 'low', 'close']].max().max())
                },
                'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
                'timeframe': detect_timeframe(df)
            }
            
            return jsonify({
                'success': True,
                'data_info': data_info,
                'message': f'Successfully loaded data, total {len(df)} rows'
            })
            
        except Exception as e:
            return jsonify({'error': 'Failed to load data'}), 500

    @app.route('/api/predict', methods=['POST'])
    @require_api_key
    def predict():
        """Perform prediction"""
        try:
            data = request.get_json()
            file_path = data.get('file_path')
            lookback = int(data.get('lookback', 400))
            pred_len = int(data.get('pred_len', 120))
            
            temperature = float(data.get('temperature', 1.0))
            top_p = float(data.get('top_p', 0.9))
            sample_count = int(data.get('sample_count', 1))
            
            if not file_path:
                return jsonify({'error': 'File path cannot be empty'}), 400
            
            df, error = load_data_file(file_path)
            if error:
                return jsonify({'error': error}), 400
            
            if len(df) < lookback:
                return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
            
            if MODEL_AVAILABLE and get_predictor() is not None:
                try:
                    required_cols = ['open', 'high', 'low', 'close']
                    if 'volume' in df.columns:
                        required_cols.append('volume')
                    
                    start_date = data.get('start_date')
                    
                    if start_date:
                        start_dt = pd.to_datetime(start_date)
                        mask = df['timestamps'] >= start_dt
                        time_range_df = df[mask]
                        
                        if len(time_range_df) < lookback + pred_len:
                            return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                        
                        x_df = time_range_df.iloc[:lookback][required_cols]
                        x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                        y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                        
                        start_timestamp = time_range_df['timestamps'].iloc[0]
                        end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                        time_span = end_timestamp - start_timestamp
                        
                        prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                    else:
                        x_df = df.iloc[:lookback][required_cols]
                        x_timestamp = df.iloc[:lookback]['timestamps']
                        y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                        prediction_type = "Kronos model prediction (latest data)"
                    
                    if isinstance(x_timestamp, pd.DatetimeIndex):
                        x_timestamp = pd.Series(x_timestamp, name='timestamps')
                    if isinstance(y_timestamp, pd.DatetimeIndex):
                        y_timestamp = pd.Series(y_timestamp, name='timestamps')
                    
                    pred_df = get_predictor().predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=pred_len,
                        T=temperature,
                        top_p=top_p,
                        sample_count=sample_count
                    )
                    
                except Exception as e:
                    return jsonify({'error': 'Prediction failed'}), 500
            else:
                return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
            
            actual_data = []
            actual_df = None
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback + pred_len:
                    actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                    
                    for i, (_, row) in enumerate(actual_df.iterrows()):
                        actual_data.append({
                            'timestamp': row['timestamps'].isoformat(),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']) if 'volume' in row else 0,
                            'amount': float(row['amount']) if 'amount' in row else 0
                        })
            else:
                if len(df) >= lookback + pred_len:
                    actual_df = df.iloc[lookback:lookback+pred_len]
                    for i, (_, row) in enumerate(actual_df.iterrows()):
                        actual_data.append({
                            'timestamp': row['timestamps'].isoformat(),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']) if 'volume' in row else 0,
                            'amount': float(row['amount']) if 'amount' in row else 0
                        })
            
            if start_date:
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
            else:
                historical_start_idx = 0
            
            chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
            
            if 'timestamps' in df.columns:
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    if len(time_range_df) >= lookback:
                        last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                        time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                        future_timestamps = pd.date_range(
                            start=last_timestamp + time_diff,
                            periods=pred_len,
                            freq=time_diff
                        )
                    else:
                        future_timestamps = []
                else:
                    last_timestamp = df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
            else:
                future_timestamps = range(len(df), len(df) + pred_len)
            
            prediction_results = []
            for i, (_, row) in enumerate(pred_df.iterrows()):
                prediction_results.append({
                    'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']) if 'volume' in row else 0,
                    'amount': float(row['amount']) if 'amount' in row else 0
                })
            
            try:
                save_prediction_results(
                    file_path=file_path,
                    prediction_type=prediction_type,
                    prediction_results=prediction_results,
                    actual_data=actual_data,
                    input_data=x_df,
                    prediction_params={
                        'lookback': lookback,
                        'pred_len': pred_len,
                        'temperature': temperature,
                        'top_p': top_p,
                        'sample_count': sample_count,
                        'start_date': start_date if start_date else 'latest'
                    }
                )
            except Exception as e:
                logger.error("Failed to save prediction results: %s", e)
            
            return jsonify({
                'success': True,
                'prediction_type': prediction_type,
                'chart': chart_json,
                'prediction_results': prediction_results,
                'actual_data': actual_data,
                'has_comparison': len(actual_data) > 0,
                'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
            })
            
        except Exception as e:
            return jsonify({'error': 'Prediction failed'}), 500

    @app.route('/api/load-model', methods=['POST'])
    @require_api_key
    def load_model():
        """Load Kronos model"""
        try:
            if not MODEL_AVAILABLE:
                return jsonify({'error': 'Kronos model library not available'}), 400
            
            data = request.get_json()
            model_key = data.get('model_key', 'kronos-small')
            device = data.get('device', 'cpu')
            
            if model_key not in AVAILABLE_MODELS:
                return jsonify({'error': f'Unsupported model: {model_key}'}), 400
            
            model_config = AVAILABLE_MODELS[model_key]
            
            _tok_rev = _get_pinned_revision(model_config['tokenizer_id'])
            _mdl_rev = _get_pinned_revision(model_config['model_id'])
            tokenizer = KronosTokenizer.from_pretrained(
                model_config['tokenizer_id'],
                **({'revision': _tok_rev} if _tok_rev else {}),
            )
            model = Kronos.from_pretrained(
                model_config['model_id'],
                **({'revision': _mdl_rev} if _mdl_rev else {}),
            )
            
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
            
            set_model_state(tokenizer, model, predictor)
            
            return jsonify({
                'success': True,
                'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
                'model_info': {
                    'name': model_config['name'],
                    'params': model_config['params'],
                    'context_length': model_config['context_length'],
                    'description': model_config['description']
                }
            })
            
        except Exception as e:
            return jsonify({'error': 'Model loading failed'}), 500

    @app.route('/api/available-models')
    def get_available_models():
        """Get available model list"""
        return jsonify({
            'models': AVAILABLE_MODELS,
            'model_available': MODEL_AVAILABLE
        })

    @app.route('/api/model-status')
    def get_model_status():
        """Get model status"""
        from .services import get_predictor as pred_ref
        _pred = pred_ref()
        if MODEL_AVAILABLE:
            if _pred is not None:
                return jsonify({
                    'available': True,
                    'loaded': True,
                    'message': 'Kronos model loaded and available',
                    'current_model': {
                        'name': _pred.model.__class__.__name__,
                        'device': str(next(_pred.model.parameters()).device)
                    }
                })
            else:
                return jsonify({
                    'available': True,
                    'loaded': False,
                    'message': 'Kronos model available but not loaded'
                })
        else:
            return jsonify({
                'available': False,
                'loaded': False,
                'message': 'Kronos model library not available, please install related dependencies'
            })
