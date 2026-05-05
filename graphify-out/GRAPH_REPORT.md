# Graph Report - Kronos  (2026-05-05)

## Corpus Check
- 67 files · ~217,407 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 777 nodes · 1200 edges · 40 communities (32 shown, 8 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 225 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]

## God Nodes (most connected - your core abstractions)
1. `make_model()` - 59 edges
2. `CustomKlineDataset` - 39 edges
3. `ConfigLoader` - 38 edges
4. `MockKronosTokenizer` - 27 edges
5. `SequentialTrainer` - 27 edges
6. `CustomFinetuneConfig` - 27 edges
7. `MockKronosBackbone` - 25 edges
8. `KronosTokenizer` - 22 edges
9. `Kronos` - 21 edges
10. `TestConfigLoader` - 21 edges

## Surprising Connections (you probably didn't know these)
- `QlibTestDataset` --uses--> `Kronos`  [INFERRED]
  finetune/qlib_test.py → model/kronos.py
- `QlibTestDataset` --uses--> `KronosTokenizer`  [INFERRED]
  finetune/qlib_test.py → model/kronos.py
- `QlibBacktest` --uses--> `Kronos`  [INFERRED]
  finetune/qlib_test.py → model/kronos.py
- `QlibBacktest` --uses--> `KronosTokenizer`  [INFERRED]
  finetune/qlib_test.py → model/kronos.py
- `generate_predictions()` --calls--> `auto_regressive_inference()`  [INFERRED]
  finetune/qlib_test.py → model/kronos.py

## Communities (40 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (13): make_model(), Tests for classification/kronos_classification_base.py — KronosClassificationMod, With loss_type=None, mock init falls through to default 'cross_entropy'., Real __init__ should wrap OSError/ConnectionError as RuntimeError., Focal loss should down-weight easy examples., Overriding non-architectural kwargs (e.g., max_context) should work.         Ove, Factory fixture that creates KronosClassificationModel with mocked backbone., TestClassificationForward (+5 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (33): Dataset, Config, Configuration class for the entire project., QlibDataset, Sets a new seed for the random sampler for each epoch. This is crucial, Whitelist-based unpickler to prevent arbitrary code execution., Returns the number of samples per epoch., Retrieves a random sample from the dataset.          Note: The `idx` argument (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (18): Function, BSQuantizer, codebook_entropy(), DependencyAwareLayer, DifferentiableEntropyFunction, DualHead, FeedForward, FixedEmbedding (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (4): ConfigLoader, CustomFinetuneConfig, TestConfigLoader, TestCustomFinetuneConfig

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (13): create_dataloaders(), CustomKlineDataset, main(), Kronos Basemodel Fine-tuning Training Script  Fine-tunes the Kronos Predictor mo, Custom dataset for K-line (candlestick) time series data.      Loads OHLCV data, setup_logging(), train_model(), Tests for finetune_csv/ modules: - CustomKlineDataset (data loading, CSV parsing (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (20): api_client(), client(), Security tests for Kronos WebUI — path traversal, auth, error sanitization, inpu, Verify @require_api_key enforces authentication on protected endpoints., Verify secrets.compare_digest is used for timing-attack resistance., Create a Flask test client with a known API key., Endpoints without @require_api_key should work without key., Verify 500 responses do not leak internal details or stack traces. (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (25): KronosClassificationModel, Pool sequence representations based on pooling strategy.                  Args:, Compute loss based on configuration.          Args:             logits: Model lo, Forward pass through the model.          Args:             input_ids: Tokenized, Tokenize time series data using Kronos tokenizer with smart padding.          Ar, Save model and tokenizer to directory.          Args:             save_directory, Kronos model with custom classification head.     Removes the original predictio, collate_fn() (+17 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (26): create_app(), Kronos WebUI — Flask app factory., Create and configure the Flask application., _load_or_generate_api_key(), Configuration, environment variables, and API key management., Load API key from file → env → generate + persist., Flask route definitions for Kronos WebUI., Decorator: require X-API-Key header for API endpoints. (+18 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (16): collate_fn(), KronosFineTuner, KronosTimeSeriesDataset, main(), Kronos Fine-tuning Script Fine-tune pretrained classification model on specific, Calculate class weights for weighted loss (inverse frequency)., Load and split data from JSON files., Collate function to pad sequences. (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (18): from_pretrained(), _get_pinned_revision(), KronosClassificationConfig, KronosClassificationONNXWrapper, Kronos Classification Model - Base Architecture Removes the original prediction, Initialize Kronos Classification Model.          Args:             kronos_model_, Return pinned commit hash for a HuggingFace model, or None for local paths., Validate checkpoint file integrity.      Args:         path: Path to checkpoint (+10 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (19): create_dataloaders(), main(), Main function to orchestrate the DDP training process., Creates and returns distributed dataloaders for training and validation., The main training and validation loop for the predictor., train_model(), TestFinetuneTokenizer, cleanup_ddp() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (5): main(), Kronos Sequential Fine-tuning Pipeline  Orchestrates sequential fine-tuning of t, Orchestrates sequential fine-tuning of Kronos Tokenizer and Predictor.      Mana, SequentialTrainer, TestSequentialTrainer

### Community 12 - "Community 12"
Cohesion: 0.16
Nodes (8): generate_output(), calc_time_stamps(), KronosPredictor, Perform parallel (batch) prediction on multiple time series. All series must hav, TestCalcTimeStamps, set_seed(), test_kronos_predictor_mse(), test_kronos_predictor_regression()

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (11): auto_regressive_inference(), Kronos, Encodes the input data into quantized indices.          Args:             x (, Kronos Model.      Args:         s1_bits (int): Number of bits for pre tokens, Args:             s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [, Decodes only the s1 tokens.          This method performs a forward pass to pr, Decodes the s2 tokens, conditioned on the context and s1 tokens.          This, predictor() (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (6): BinarySphericalQuantizer, Converts a `code` to an index in the codebook.         Args:             zhat:, Converts a `code` to a list of indexes (in groups) in the codebook.         Arg, Inverse of `indexes_to_codes`., Inverse of `group_indexes_to_codes`., Paper link: https://arxiv.org/pdf/2406.07548.pdf         Here we use the offici

### Community 15 - "Community 15"
Cohesion: 0.18
Nodes (8): main(), PolicyGradientFinetuner, Kronos RL Fine-tuning Script Fine-tune classification model using Reinforcement, Setup training and validation data loaders., Setup optimizer and learning rate scheduler., Compute reward for each action (prediction).          Args:             logit, Compute policy gradient loss (REINFORCE).          Args:             logits:, Fine-tune using REINFORCE (Policy Gradient) algorithm.      This treats classi

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (16): _force_cpu(), mock_tokenizer(), Shared test fixtures and configuration for Kronos test suite.  All external depe, DataFrame with standard OHLCV+amount columns (50 rows)., DataFrame with only OHLC columns., DataFrame with OHLCV + timestamps (200 rows), suitable for finetune_csv tests., Return a temporary directory path., Create a JSON file mimicking the classification training data format. (+8 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (7): Filter a distribution of logits using top-k and/or nucleus (top-p) filtering, Sample tokens from logits with temperature and top-k/top-p filtering.      Arg, sample_from_logits(), top_k_top_p_filtering(), top_k=0 skips the top-k branch; no top_p branch entered either -> returns None., top_p=1.0 skips top-p branch, function returns None., TestSamplingFunctions

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (12): analyze_checkpoint(), convert_csv_to_classification_data(), create_sample_data(), KronosClassificationPipeline, Kronos Inference and Utility Scripts Includes inference pipeline and helper util, Inference pipeline for Kronos classification model., Predict on data from file and save results.          Args:             input_fil, Create sample time series data for testing.     Demonstrates the expected data f (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (3): Test KronosTokenizer module directly (small dims)., TestKronosTokenizer, tokenizer()

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (10): finetuner_setup(), _make_classification_model(), pretrainer_setup(), Tests for classification/kronos_pretrain.py and classification/kronos_finetune.p, Create a directory with multiple JSON files., Create a KronosClassificationModel with mocked HF downloads., Create a JSON file with classification training data., sample_json_data() (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (6): mock_backbone(), _MockConfig, MockKronosBackbone, Minimal config object that mimics Kronos backbone config., Tiny stand-in for the Kronos model that produces hidden_states., TestClassificationConfig

### Community 23 - "Community 23"
Cohesion: 0.21
Nodes (3): model(), Test Kronos model forward pass., TestKronosModel

### Community 25 - "Community 25"
Cohesion: 0.2
Nodes (7): MockKronosTokenizer, Minimal tokenizer stand-in., _mock_init(), Replacement __init__ that skips HF downloads., _mock_classification_init(), mock_tokenizer(), Replacement __init__ that skips HF downloads.

### Community 26 - "Community 26"
Cohesion: 0.25
Nodes (6): KronosTokenizer, Converts indices to bit representations and scales them.          Args:, KronosTokenizer module for tokenizing input data using a hybrid quantization app, Decodes quantized indices back to the input data space.          Args:, Forward pass of the KronosTokenizer.          Args:             x (torch.Tens, PyTorchModelHubMixin

### Community 28 - "Community 28"
Cohesion: 0.6
Nodes (5): apply_price_limits(), load_data(), plot_result(), predict_future(), prepare_inputs()

### Community 31 - "Community 31"
Cohesion: 0.5
Nodes (3): Shared fixture definitions that need to be available across test modules. This i, Create a minimal YAML config file for finetune_csv tests., sample_yaml_config()

## Knowledge Gaps
- **174 isolated node(s):** `A class to handle the loading, processing, and splitting of Qlib financial data.`, `Initializes the preprocessor with configuration and data fields.`, `Initializes the Qlib environment.`, `Loads raw data from Qlib, processes it symbol by symbol, and stores         it`, `Splits the loaded data into train, validation, and test sets and saves them to d` (+169 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Kronos` connect `Community 13` to `Community 1`, `Community 2`, `Community 4`, `Community 11`, `Community 12`, `Community 17`, `Community 19`, `Community 22`, `Community 23`, `Community 26`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `KronosTokenizer` connect `Community 26` to `Community 1`, `Community 2`, `Community 4`, `Community 11`, `Community 12`, `Community 13`, `Community 17`, `Community 19`, `Community 22`, `Community 23`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `QlibTestDataset` connect `Community 1` to `Community 26`, `Community 13`?**
  _High betweenness centrality (0.177) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `CustomKlineDataset` (e.g. with `TestCustomKlineDatasetLoading` and `TestCustomKlineDatasetSplitting`) actually correct?**
  _`CustomKlineDataset` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `ConfigLoader` (e.g. with `TestCustomKlineDatasetLoading` and `TestCustomKlineDatasetSplitting`) actually correct?**
  _`ConfigLoader` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `MockKronosTokenizer` (e.g. with `TestKronosTimeSeriesDataset` and `TestCollateFn`) actually correct?**
  _`MockKronosTokenizer` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `SequentialTrainer` (e.g. with `TestCustomKlineDatasetLoading` and `TestCustomKlineDatasetSplitting`) actually correct?**
  _`SequentialTrainer` has 16 INFERRED edges - model-reasoned connections that need verification._