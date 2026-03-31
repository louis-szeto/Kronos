# HARNESS.md — Kronos

## Project Overview

Time-series foundation model for financial markets. BSQ tokenization + dual-head Transformer (autoregressive + classification). 35 Python files, 201 tests. Training pipeline uses PyTorch + HuggingFace.

**Critical invariant:** Training data never leaves controlled AWS infrastructure.

## CRITICAL FINDINGS

### 🔴 Security Issues (P0)

1. **WebUI (Flask)** — potential attack surface
   - Input validation on prediction requests
   - CORS configuration
   - Rate limiting
   - No auth on prediction endpoint?

2. **Model artifacts** — verify S3 access controls for saved models
   - No sensitive data embedded in model weights
   - Access restricted to authorized services

3. **Training data pipeline** — verify:
   - Data sources are validated (OHLCV constraints)
   - No data leakage between train/test splits
   - Checkpoint encryption at rest

### 🟡 Stability (P1)
- Multi-GPU training resilience
- Checkpoint recovery after crash
- Memory management during training (gradient accumulation, mixed precision)
- Walk-forward validation correctness

### 🟢 Performance
- BSQ quantization efficiency
- Batch prediction throughput
- Model inference latency

## Refactor Priority

### Wave 1: Security
- [ ] Audit Flask WebUI for injection, CORS, auth
- [ ] Verify S3 ACLs on model artifacts
- [ ] Add input validation on all prediction endpoints
- [ ] Verify no data leakage in walk-forward splits

### Wave 2: Test Coverage to 95%
- [ ] Integration tests for training pipeline
- [ ] Model serialization/deserialization tests
- [ ] WebUI endpoint tests

### Wave 3: Stability & Cost
- [ ] Multi-GPU failure recovery
- [ ] Memory profiling during training
- [ ] GPU utilization optimization
- [ ] Spot instance support for training jobs

## Invariants

1. Training data never leaves AWS
2. All prediction endpoints validate input
3. Model artifacts encrypted at rest
4. No data leakage between train/test
5. WebUI requires authentication
