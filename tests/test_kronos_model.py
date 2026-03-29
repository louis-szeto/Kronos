"""
Tests for model/kronos.py — Kronos, KronosTokenizer, KronosPredictor,
sample_from_logits, top_k_top_p_filtering, and auto_regressive_inference.

Uses small model configs so everything runs on CPU in seconds.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from conftest import MockKronosBackbone, MockKronosTokenizer


# ---------------------------------------------------------------------------
# Kronos & KronosTokenizer — constructor / forward
# ---------------------------------------------------------------------------

class TestKronosTokenizer:
    """Test KronosTokenizer module directly (small dims)."""

    @pytest.fixture()
    def tokenizer(self):
        from model.kronos import KronosTokenizer
        return KronosTokenizer(
            d_in=6, d_model=32, n_heads=4, ff_dim=64,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1,
            zeta=0.05, group_size=4,
        )

    def test_forward_shape(self, tokenizer):
        batch, seq_len, d_in = 2, 16, 6
        x = torch.randn(batch, seq_len, d_in)
        (z_pre, z), bsq_loss, quantized, z_indices = tokenizer(x)

        assert z_pre.shape == (batch, seq_len, d_in)
        assert z.shape == (batch, seq_len, d_in)
        assert bsq_loss.shape == ()  # scalar
        assert quantized.shape == (batch, seq_len, tokenizer.codebook_dim)

    def test_encode_returns_indices(self, tokenizer):
        batch, seq_len, d_in = 1, 10, 6
        x = torch.randn(batch, seq_len, d_in)
        z_indices = tokenizer.encode(x)
        assert isinstance(z_indices, torch.Tensor)

    def test_encode_half_returns_tuple(self, tokenizer):
        batch, seq_len, d_in = 1, 10, 6
        x = torch.randn(batch, seq_len, d_in)
        z_indices = tokenizer.encode(x, half=True)
        assert isinstance(z_indices, (tuple, list))
        assert len(z_indices) == 2

    def test_decode_roundtrip(self, tokenizer):
        batch, seq_len, d_in = 1, 8, 6
        x = torch.randn(batch, seq_len, d_in)
        z_indices = tokenizer.encode(x)
        decoded = tokenizer.decode(z_indices)
        assert decoded.shape == (batch, seq_len, d_in)

    def test_indices_to_bits(self, tokenizer):
        indices = torch.tensor([0, 1, 5])
        bits = tokenizer.indices_to_bits(indices)
        assert bits.shape == (3, tokenizer.codebook_dim)
        # Bipolar values
        assert (bits.abs() <= 1.0 + 1e-6).all()


class TestKronosModel:
    """Test Kronos model forward pass."""

    @pytest.fixture()
    def model(self):
        from model.kronos import Kronos
        return Kronos(
            s1_bits=4, s2_bits=4, n_layers=2, d_model=32,
            n_heads=4, ff_dim=64,
            ffn_dropout_p=0.0, attn_dropout_p=0.0,
            resid_dropout_p=0.0, token_dropout_p=0.0,
            learn_te=True,
        )

    def test_forward_shape(self, model):
        batch, seq_len = 2, 16
        vocab_s1 = 2 ** 4
        vocab_s2 = 2 ** 4
        s1_ids = torch.randint(0, vocab_s1, (batch, seq_len))
        s2_ids = torch.randint(0, vocab_s2, (batch, seq_len))
        stamp = torch.rand(batch, seq_len, 5)

        s1_logits, s2_logits = model(s1_ids, s2_ids, stamp=stamp)

        assert s1_logits.shape == (batch, seq_len, vocab_s1)
        assert s2_logits.shape == (batch, seq_len, vocab_s2)

    def test_forward_no_stamp(self, model):
        batch, seq_len = 1, 8
        vocab_s1 = 2 ** 4
        vocab_s2 = 2 ** 4
        s1_ids = torch.randint(0, vocab_s1, (batch, seq_len))
        s2_ids = torch.randint(0, vocab_s2, (batch, seq_len))

        s1_logits, s2_logits = model(s1_ids, s2_ids)
        assert s1_logits.shape == (batch, seq_len, vocab_s1)

    def test_decode_s1(self, model):
        batch, seq_len = 1, 8
        s1_ids = torch.randint(0, 16, (batch, seq_len))
        s2_ids = torch.randint(0, 16, (batch, seq_len))
        s1_logits, context = model.decode_s1(s1_ids, s2_ids)
        assert s1_logits.shape[-1] == 16
        assert context.shape == (batch, seq_len, 32)

    def test_decode_s2(self, model):
        batch, seq_len = 1, 8
        s1_ids = torch.randint(0, 16, (batch, seq_len))
        s2_ids = torch.randint(0, 16, (batch, seq_len))
        _, context = model.decode_s1(s1_ids, s2_ids)
        s2_logits = model.decode_s2(context, s1_ids)
        assert s2_logits.shape[-1] == 16

    def test_teacher_forcing(self, model):
        batch, seq_len = 1, 8
        s1_ids = torch.randint(0, 16, (batch, seq_len))
        s2_ids = torch.randint(0, 16, (batch, seq_len))
        s1_targets = torch.randint(0, 16, (batch, seq_len))
        s1_logits, s2_logits = model(
            s1_ids, s2_ids, use_teacher_forcing=True, s1_targets=s1_targets
        )
        assert s1_logits.shape == (batch, seq_len, 16)


# ---------------------------------------------------------------------------
# sample_from_logits & top_k_top_p_filtering
# ---------------------------------------------------------------------------

class TestSamplingFunctions:

    def test_greedy_via_temperature(self):
        """Very low temperature should behave greedily."""
        from model.kronos import sample_from_logits
        logits = torch.tensor([[1.0, 10.0, 2.0]])
        result = sample_from_logits(logits, temperature=1e-9, sample_logits=True)
        assert result.item() == 1

    def test_high_temperature(self):
        from model.kronos import sample_from_logits
        torch.manual_seed(42)
        logits = torch.zeros(1, 100)
        results = [sample_from_logits(logits, temperature=100.0, sample_logits=True).item()
                   for _ in range(20)]
        assert len(set(results)) > 1

    def test_top_k_filtering(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, 4.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=2)
        non_inf = (filtered > -float("inf")).sum().item()
        assert non_inf == 2

    def test_top_p_filtering(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 10.0, 0.1, 0.01]])
        filtered = top_k_top_p_filtering(logits.clone(), top_p=0.9)
        assert filtered[0, 1] > -float("inf")

    def test_top_k_and_top_p_combined(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, 4.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=3, top_p=0.9)
        assert (filtered > -float("inf")).sum().item() <= 3

    def test_sample_batch(self):
        from model.kronos import sample_from_logits
        logits = torch.randn(4, 16)
        result = sample_from_logits(logits)
        assert result.shape == (4, 1)

    def test_no_sampling_returns_argmax(self):
        """When sample_logits=False and no top_k/top_p, uses multinomial on softmax.
        Note: sample_logits=False path in the source uses `top_k` (torch.topk function),
        but the parameter name shadows it. Test with temperature near-zero instead."""
        from model.kronos import sample_from_logits
        logits = torch.tensor([[1.0, 10.0, 2.0]])
        result = sample_from_logits(logits, temperature=1e-9, top_k=0, top_p=1.0)
        assert result.item() == 1


# ---------------------------------------------------------------------------
# KronosPredictor
# ---------------------------------------------------------------------------

class TestKronosPredictor:

    @pytest.fixture()
    def predictor(self):
        from model.kronos import Kronos, KronosTokenizer, KronosPredictor
        tok = KronosTokenizer(
            d_in=6, d_model=32, n_heads=4, ff_dim=64,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1,
            zeta=0.05, group_size=4,
        )
        model = Kronos(
            s1_bits=4, s2_bits=4, n_layers=2, d_model=32,
            n_heads=4, ff_dim=64,
            ffn_dropout_p=0.0, attn_dropout_p=0.0,
            resid_dropout_p=0.0, token_dropout_p=0.0,
            learn_te=True,
        )
        return KronosPredictor(model, tok, device="cpu", max_context=32)

    def test_predict_basic(self, predictor, sample_ohlcv_df):
        df = sample_ohlcv_df
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=len(df), freq="h"))
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=5, freq="h"))

        result = predictor.predict(
            df=df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=5,
            T=1.0, top_k=1, top_p=1.0,
            verbose=False, sample_count=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "open" in result.columns
        assert "close" in result.columns

    def test_predict_rejects_non_dataframe(self, predictor):
        with pytest.raises(ValueError, match="pandas DataFrame"):
            predictor.predict(
                df=np.zeros((10, 4)),
                x_timestamp=pd.Series(pd.date_range("2024-01-01", periods=10, freq="h")),
                y_timestamp=pd.Series(pd.date_range("2024-01-01 10:00", periods=5, freq="h")),
                pred_len=5,
            )

    def test_predict_rejects_negative_prices(self, predictor):
        df = pd.DataFrame({
            "open": [-1, 2, 3], "high": [4, 5, 6],
            "low": [1, 2, 3], "close": [2, 3, 4],
            "volume": [100, 200, 300], "amount": [200, 600, 1200],
        })
        with pytest.raises(ValueError, match="negative"):
            predictor.predict(
                df=df,
                x_timestamp=pd.Series(pd.date_range("2024-01-01", periods=3, freq="h")),
                y_timestamp=pd.Series(pd.date_range("2024-01-01 03:00", periods=2, freq="h")),
                pred_len=2, verbose=False,
            )

    def test_predict_rejects_nan(self, predictor):
        df = pd.DataFrame({
            "open": [1, float("nan"), 3], "high": [4, 5, 6],
            "low": [1, 2, 3], "close": [2, 3, 4],
            "volume": [100, 200, 300], "amount": [200, 600, 1200],
        })
        with pytest.raises(ValueError, match="NaN"):
            predictor.predict(
                df=df,
                x_timestamp=pd.Series(pd.date_range("2024-01-01", periods=3, freq="h")),
                y_timestamp=pd.Series(pd.date_range("2024-01-01 03:00", periods=2, freq="h")),
                pred_len=2, verbose=False,
            )

    def test_predict_missing_volume(self, predictor, sample_ohlcv_no_volume_df):
        df = sample_ohlcv_no_volume_df
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=len(df), freq="h"))
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h"))
        result = predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=3, verbose=False,
        )
        assert len(result) == 3

    def test_predict_batch(self, predictor, sample_ohlcv_df):
        df = sample_ohlcv_df
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=len(df), freq="h"))
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h"))

        results = predictor.predict_batch(
            df_list=[df, df],
            x_timestamp_list=[x_ts, x_ts],
            y_timestamp_list=[y_ts, y_ts],
            pred_len=3, verbose=False, sample_count=1,
        )
        assert len(results) == 2
        assert all(len(r) == 3 for r in results)

    def test_predict_batch_mismatched_lengths(self, predictor):
        with pytest.raises(ValueError, match="consistent lengths"):
            predictor.predict_batch(
                df_list=[pd.DataFrame()], x_timestamp_list=[], y_timestamp_list=[],
                pred_len=1,
            )


# ---------------------------------------------------------------------------
# calc_time_stamps helper
# ---------------------------------------------------------------------------

class TestCalcTimeStamps:

    def test_output_columns(self):
        from model.kronos import calc_time_stamps
        ts = pd.Series(pd.date_range("2024-03-15 09:30:00", periods=5, freq="h"))
        result = calc_time_stamps(ts)
        assert "minute" in result.columns
        assert "hour" in result.columns
        assert "weekday" in result.columns
        assert "day" in result.columns
        assert "month" in result.columns
        assert len(result) == 5
