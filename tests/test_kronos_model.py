"""
Tests for model/kronos.py — Kronos, KronosTokenizer, KronosPredictor,
sample_from_logits, top_k_top_p_filtering, auto_regressive_inference.

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
# KronosTokenizer — constructor, forward, encode, decode, indices_to_bits
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

    # --- init ---

    def test_init_attributes(self, tokenizer):
        assert tokenizer.d_in == 6
        assert tokenizer.d_model == 32
        assert tokenizer.s1_bits == 4
        assert tokenizer.s2_bits == 4
        assert tokenizer.codebook_dim == 8  # s1_bits + s2_bits

    def test_init_layers(self, tokenizer):
        assert len(tokenizer.encoder) == 1   # n_enc_layers - 1
        assert len(tokenizer.decoder) == 1   # n_dec_layers - 1
        assert isinstance(tokenizer.embed, nn.Linear)
        assert isinstance(tokenizer.head, nn.Linear)
        assert isinstance(tokenizer.tokenizer, nn.Module)  # BSQuantizer

    # --- forward ---

    def test_forward_shape(self, tokenizer):
        batch, seq_len, d_in = 2, 16, 6
        x = torch.randn(batch, seq_len, d_in)
        (z_pre, z), bsq_loss, quantized, z_indices = tokenizer(x)

        assert z_pre.shape == (batch, seq_len, d_in)
        assert z.shape == (batch, seq_len, d_in)
        assert bsq_loss.shape == ()
        assert quantized.shape == (batch, seq_len, tokenizer.codebook_dim)

    def test_forward_loss_finite(self, tokenizer):
        x = torch.randn(2, 8, 6)
        _, bsq_loss, _, _ = tokenizer(x)
        assert torch.isfinite(bsq_loss)

    def test_forward_gradient_flow(self, tokenizer):
        x = torch.randn(1, 4, 6, requires_grad=True)
        (z_pre, z), bsq_loss, _, _ = tokenizer(x)
        total_loss = bsq_loss + z_pre.mean()
        total_loss.backward()
        assert x.grad is not None

    # --- encode ---

    def test_encode_returns_indices(self, tokenizer):
        x = torch.randn(1, 10, 6)
        z_indices = tokenizer.encode(x)
        assert isinstance(z_indices, torch.Tensor)

    def test_encode_half_returns_tuple(self, tokenizer):
        x = torch.randn(1, 10, 6)
        z_indices = tokenizer.encode(x, half=True)
        assert isinstance(z_indices, (tuple, list))
        assert len(z_indices) == 2

    # --- decode ---

    def test_decode_roundtrip(self, tokenizer):
        x = torch.randn(1, 8, 6)
        z_indices = tokenizer.encode(x)
        decoded = tokenizer.decode(z_indices)
        assert decoded.shape == (1, 8, 6)

    def test_decode_half_roundtrip(self, tokenizer):
        x = torch.randn(1, 8, 6)
        z_indices = tokenizer.encode(x, half=True)
        decoded = tokenizer.decode(z_indices, half=True)
        assert decoded.shape == (1, 8, 6)

    # --- indices_to_bits ---

    def test_indices_to_bits_shape(self, tokenizer):
        indices = torch.tensor([0, 1, 5])
        bits = tokenizer.indices_to_bits(indices)
        assert bits.shape == (3, tokenizer.codebook_dim)

    def test_indices_to_bits_bipolar(self, tokenizer):
        indices = torch.tensor([0, 1])
        bits = tokenizer.indices_to_bits(indices)
        # Values should be in [-1, 1] range (bipolar)
        assert (bits.abs() <= 1.0 + 1e-6).all()

    def test_indices_to_bits_half(self, tokenizer):
        idx0 = torch.tensor([0, 1, 2])
        idx1 = torch.tensor([3, 4, 5])
        bits = tokenizer.indices_to_bits((idx0, idx1), half=True)
        assert bits.shape == (3, tokenizer.codebook_dim)


# ---------------------------------------------------------------------------
# Kronos model — constructor, forward, decode_s1, decode_s2, teacher forcing
# ---------------------------------------------------------------------------

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

    # --- init ---

    def test_init_attributes(self, model):
        assert model.s1_bits == 4
        assert model.s2_bits == 4
        assert model.d_model == 32
        assert model.n_layers == 2
        assert model.n_heads == 4
        assert model.s1_vocab_size == 16  # 2 ** s1_bits

    def test_init_layers(self, model):
        assert len(model.transformer) == 2
        assert isinstance(model.embedding, nn.Module)
        assert isinstance(model.head, nn.Module)
        assert isinstance(model.dep_layer, nn.Module)

    def test_init_weights(self, model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert module.weight is not None
                if module.bias is not None:
                    assert (module.bias == 0).all()

    # --- forward ---

    def test_forward_shape(self, model):
        batch, seq_len = 2, 16
        vocab = 2 ** 4
        s1 = torch.randint(0, vocab, (batch, seq_len))
        s2 = torch.randint(0, vocab, (batch, seq_len))
        stamp = torch.rand(batch, seq_len, 5)

        s1_logits, s2_logits = model(s1, s2, stamp=stamp)
        assert s1_logits.shape == (batch, seq_len, vocab)
        assert s2_logits.shape == (batch, seq_len, vocab)

    def test_forward_no_stamp(self, model):
        batch, seq_len = 1, 8
        vocab = 2 ** 4
        s1 = torch.randint(0, vocab, (batch, seq_len))
        s2 = torch.randint(0, vocab, (batch, seq_len))

        s1_logits, s2_logits = model(s1, s2)
        assert s1_logits.shape == (batch, seq_len, vocab)

    def test_forward_with_padding_mask(self, model):
        batch, seq_len = 2, 8
        vocab = 2 ** 4
        s1 = torch.randint(0, vocab, (batch, seq_len))
        s2 = torch.randint(0, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        mask[0, 5:] = False

        s1_logits, s2_logits = model(s1, s2, padding_mask=mask)
        assert s1_logits.shape == (batch, seq_len, vocab)

    # --- decode_s1 / decode_s2 ---

    def test_decode_s1(self, model):
        batch, seq_len = 1, 8
        s1 = torch.randint(0, 16, (batch, seq_len))
        s2 = torch.randint(0, 16, (batch, seq_len))
        s1_logits, context = model.decode_s1(s1, s2)
        assert s1_logits.shape[-1] == 16
        assert context.shape == (batch, seq_len, 32)

    def test_decode_s2(self, model):
        batch, seq_len = 1, 8
        s1 = torch.randint(0, 16, (batch, seq_len))
        s2 = torch.randint(0, 16, (batch, seq_len))
        _, context = model.decode_s1(s1, s2)
        s2_logits = model.decode_s2(context, s1)
        assert s2_logits.shape[-1] == 16

    # --- teacher forcing ---

    def test_teacher_forcing(self, model):
        batch, seq_len = 1, 8
        s1 = torch.randint(0, 16, (batch, seq_len))
        s2 = torch.randint(0, 16, (batch, seq_len))
        targets = torch.randint(0, 16, (batch, seq_len))
        s1_logits, s2_logits = model(s1, s2, use_teacher_forcing=True, s1_targets=targets)
        assert s1_logits.shape == (batch, seq_len, 16)
        assert s2_logits.shape == (batch, seq_len, 16)


# ---------------------------------------------------------------------------
# sample_from_logits & top_k_top_p_filtering
# ---------------------------------------------------------------------------

class TestSamplingFunctions:

    # --- sample_from_logits ---

    def test_greedy_via_low_temperature(self):
        from model.kronos import sample_from_logits
        logits = torch.tensor([[1.0, 10.0, 2.0]])
        result = sample_from_logits(logits, temperature=1e-9, sample_logits=True)
        assert result.item() == 1  # argmax

    def test_high_temperature_diversity(self):
        from model.kronos import sample_from_logits
        torch.manual_seed(42)
        logits = torch.zeros(1, 100)
        results = [sample_from_logits(logits, temperature=100.0, sample_logits=True).item()
                   for _ in range(20)]
        assert len(set(results)) > 1

    def test_sample_batch_shape(self):
        from model.kronos import sample_from_logits
        logits = torch.randn(4, 16)
        result = sample_from_logits(logits)
        assert result.shape == (4, 1)

    def test_sample_output_in_range(self):
        from model.kronos import sample_from_logits
        logits = torch.randn(2, 10)
        result = sample_from_logits(logits, temperature=1.0)
        assert (result >= 0).all()
        assert (result < 10).all()

    # --- top_k_top_p_filtering ---

    def test_top_k_filters_to_k(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, 4.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=2)
        non_inf = (filtered > -float("inf")).sum().item()
        assert non_inf == 2

    def test_top_k_zero_returns_none(self):
        """top_k=0 skips the top-k branch; no top_p branch entered either -> returns None."""
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=0)
        assert filtered is None

    def test_top_p_nucleus(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 10.0, 0.1, 0.01]])
        filtered = top_k_top_p_filtering(logits.clone(), top_p=0.9)
        assert filtered[0, 1] > -float("inf")

    def test_top_p_one_returns_none(self):
        """top_p=1.0 skips top-p branch, function returns None."""
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_p=1.0)
        assert filtered is None

    def test_top_k_and_top_p_combined(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5, 4.0]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=3, top_p=0.9)
        assert (filtered > -float("inf")).sum().item() <= 3

    def test_min_tokens_to_keep(self):
        from model.kronos import top_k_top_p_filtering
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
        filtered = top_k_top_p_filtering(logits.clone(), top_k=1, min_tokens_to_keep=2)
        non_inf = (filtered > -float("inf")).sum().item()
        assert non_inf >= 2


# ---------------------------------------------------------------------------
# auto_regressive_inference
# ---------------------------------------------------------------------------

class TestAutoRegressiveInference:

    @pytest.fixture()
    def tokenizer_model(self):
        from model.kronos import KronosTokenizer, Kronos
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
        return tok, model

    def test_ar_inference_output_shape(self, tokenizer_model):
        from model.kronos import auto_regressive_inference
        tok, model = tokenizer_model
        batch = 1
        seq_len = 10
        pred_len = 3
        x = torch.randn(batch, seq_len, 6)
        x_stamp = torch.rand(batch, seq_len, 5)
        y_stamp = torch.rand(batch, pred_len, 5)

        preds = auto_regressive_inference(
            tok, model, x, x_stamp, y_stamp,
            max_context=32, pred_len=pred_len,
            clip=5.0, T=1.0, top_k=0, top_p=1.0,
            sample_count=1, verbose=False,
        )
        # Returns full decoded sequence: (batch, seq_len + pred_len, d_in)
        assert preds.shape[0] == batch
        assert preds.shape[2] == 6  # d_in
        assert preds.shape[1] >= pred_len

    def test_ar_inference_multiple_samples(self, tokenizer_model):
        from model.kronos import auto_regressive_inference
        tok, model = tokenizer_model
        x = torch.randn(1, 8, 6)
        x_stamp = torch.rand(1, 8, 5)
        y_stamp = torch.rand(1, 2, 5)

        preds = auto_regressive_inference(
            tok, model, x, x_stamp, y_stamp,
            max_context=32, pred_len=2,
            sample_count=3, verbose=False,
        )
        # Averaged over 3 samples, returns full decoded sequence
        assert preds.shape[0] == 1
        assert preds.shape[2] == 6
        assert preds.shape[1] >= 2


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

    def test_values_correct(self):
        from model.kronos import calc_time_stamps
        ts = pd.Series([pd.Timestamp("2024-03-15 09:30:00")])
        result = calc_time_stamps(ts)
        assert result["minute"].iloc[0] == 30
        assert result["hour"].iloc[0] == 9
        assert result["weekday"].iloc[0] == 4  # Friday
        assert result["day"].iloc[0] == 15
        assert result["month"].iloc[0] == 3


# ---------------------------------------------------------------------------
# KronosPredictor — predict, predict_batch, input validation
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

    # --- predict ---

    def test_predict_basic(self, predictor, sample_ohlcv_df):
        df = sample_ohlcv_df
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=len(df), freq="h"))
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=5, freq="h"))

        result = predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=5, T=1.0, top_k=1, top_p=1.0,
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

    def test_predict_rejects_missing_price_columns(self, predictor):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Price columns"):
            predictor.predict(
                df=df,
                x_timestamp=pd.Series(pd.date_range("2024-01-01", periods=1, freq="h")),
                y_timestamp=pd.Series(pd.date_range("2024-01-01 01:00", periods=1, freq="h")),
                pred_len=1, verbose=False,
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

    def test_predict_rejects_inf(self, predictor):
        df = pd.DataFrame({
            "open": [1, float("inf"), 3], "high": [4, 5, 6],
            "low": [1, 2, 3], "close": [2, 3, 4],
            "volume": [100, 200, 300], "amount": [200, 600, 1200],
        })
        with pytest.raises(ValueError, match="infinite"):
            predictor.predict(
                df=df,
                x_timestamp=pd.Series(pd.date_range("2024-01-01", periods=3, freq="h")),
                y_timestamp=pd.Series(pd.date_range("2024-01-01 03:00", periods=2, freq="h")),
                pred_len=2, verbose=False,
            )

    def test_predict_fills_missing_volume(self, predictor, sample_ohlcv_no_volume_df):
        df = sample_ohlcv_no_volume_df
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=len(df), freq="h"))
        y_ts = pd.Series(pd.date_range(x_ts.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h"))
        result = predictor.predict(
            df=df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=3, verbose=False,
        )
        assert len(result) == 3

    # --- predict_batch ---

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

    def test_predict_batch_rejects_non_list(self, predictor):
        with pytest.raises(ValueError, match="list or tuple"):
            predictor.predict_batch(
                df_list="not_a_list",
                x_timestamp_list=[],
                y_timestamp_list=[],
                pred_len=1,
            )

    def test_predict_batch_rejects_non_df_items(self, predictor):
        with pytest.raises(ValueError, match="not a pandas DataFrame"):
            predictor.predict_batch(
                df_list=["not_a_df"],
                x_timestamp_list=[pd.Series(pd.date_range("2024-01-01", periods=1, freq="h"))],
                y_timestamp_list=[pd.Series(pd.date_range("2024-01-01 01:00", periods=1, freq="h"))],
                pred_len=1,
            )

    def test_predict_batch_inconsistent_seq_lens(self, predictor, sample_ohlcv_df):
        df_short = sample_ohlcv_df.iloc[:10]
        df_long = sample_ohlcv_df.iloc[:20]
        x_ts_s = pd.Series(pd.date_range("2024-01-01", periods=10, freq="h"))
        x_ts_l = pd.Series(pd.date_range("2024-01-01", periods=20, freq="h"))
        y_ts = pd.Series(pd.date_range("2024-01-02", periods=2, freq="h"))

        with pytest.raises(ValueError, match="consistent historical lengths"):
            predictor.predict_batch(
                df_list=[df_short, df_long],
                x_timestamp_list=[x_ts_s, x_ts_l],
                y_timestamp_list=[y_ts, y_ts],
                pred_len=2, verbose=False,
            )
