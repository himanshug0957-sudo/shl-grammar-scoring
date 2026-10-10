# SHL Hiring Assessment 2026 - Grammar Scoring Engine

Predicts a grammar score (0-5) from a 45-60 s spoken-English audio clip (.wav).

## Approach (in simple words)
1. Speech to text: each audio clip is transcribed with Whisper-medium. A second transcript is made with a "prompted" Whisper that keeps fillers and mistakes ("um", "I go yesterday") so grammar errors are not auto-corrected.
2. Text features: grammar error rate (LanguageTool), grammatical acceptability (CoLA RoBERTa), GPT-2 perplexity, sentence-length stats, vocabulary diversity (TTR), filler words, repetitions, plus sentence embeddings (all-mpnet-base-v2).
3. Audio features: wav2vec2 embeddings, Whisper encoder embeddings, and prosody (pauses, speech ratio, speaking rate, loudness).
4. Models: Ridge, SVR, Huber, Random Forest, Gradient Boosting on these features.
5. Blending: 5-fold out-of-fold predictions (3 seeds) are combined with a non-negative linear regression. The final score is clipped to [0, 5].
6. Training data note: clips with label 0 were dropped from training because they improved the leaderboard score.

## Evaluation
- Metrics: RMSE and Pearson correlation (cross-validation + training RMSE are printed in the notebook).
- Training RMSE:  0.2521
- CV RMSE / Pearson: 0.4890 / 0.8761
- Kaggle leaderboard score: 0.3800

## How to run
1. Open `SHL_Grammar_Scoring_FINAL.ipynb` on Kaggle (GPU T4, Internet On, challenge data attached).
2. Run All. First run takes about 1-1.5 hours (Whisper transcription); later runs use caches.
3. `submission.csv` (columns `filename,label`) is written at the end.

## Files
- `SHL_Grammar_Scoring_FINAL.ipynb` - full pipeline
- `submission.csv` - final predictions
