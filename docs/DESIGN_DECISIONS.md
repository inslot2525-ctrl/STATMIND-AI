# Design decisions

Why StatMind is built the way it is: each decision, the reason, the alternative that was not
chosen, and where the code lives. The questions at the end of each part are the ones a reviewer
is most likely to ask.

## 1. Model evaluation

### The test split is set aside before anything else
`prepare_ml_data` in `backend/services/ml_engine.py` splits the data first (80/20 by default,
stratified for classification, seed 42). Cross-validation, tuning and model selection only ever
see the training part. The test part is scored once, by the winner.

**Why:** if you pick the best of eight models using the test set, the test score is no longer an
honest estimate; you have selected for luck on those rows. Keeping it untouched means the headline
number is an unbiased estimate of performance on new data.

### Models are ranked by 5-fold cross-validation
`_cv_splitter` uses `StratifiedKFold(5, shuffle=True)` for classification and `KFold` for
regression. The leaderboard shows mean ± standard deviation.

**Why:** one train/validation split gives a noisy number, and on a few hundred rows the ranking can
flip with the seed. Five folds average that out, and the standard deviation shows when two models
are effectively tied. **Why 5 and not 10:** a common bias/variance compromise, and half the compute.
**Stratified** keeps the class mix the same in every fold, which matters for small or imbalanced data.

### Ranking metric: macro F1 for classification, R² for regression
`RANKING_METRIC` in `ml_engine.py`.

**Why macro F1:** accuracy and weighted F1 are dominated by the majority class. On a 90/10 target a
model that never predicts the minority class scores 0.90 accuracy. Macro F1 averages the per-class
F1 scores equally, so a model has to do well on every class to rank high.
**Trade-off:** macro F1 depends on the decision threshold; ROC AUC does not, and is reported
alongside for that reason.

### Only the top two models are tuned, with a random search
`_tune_candidate`: `RandomizedSearchCV`, up to 10 configurations, same folds.

**Why random, not grid:** with several hyperparameters, a grid spends most of its budget on values of
unimportant parameters; random search covers the important ones better for the same cost.
**Why only the top two:** tuning everything multiplies training time for models that were never
close. **Why not Bayesian optimisation:** another dependency and more moving parts for a small gain
at this budget. If the defaults score better than every sampled configuration, the defaults are kept.

**The honest caveat:** choosing the best of ~10 configurations on the same folds makes the winner's
CV score slightly optimistic. That is exactly why the final number comes from the untouched test split.

### Every score sits next to a baseline
`get_baseline_model`: `DummyClassifier(strategy="prior")` or `DummyRegressor(strategy="mean")`,
shown as a leaderboard row and under each metric tile.

**Why:** a score means nothing alone. 0.81 accuracy is good if always guessing the majority class
gets 0.69, and worthless if it gets 0.80.

**Likely questions**
- *Is the reported test score biased by tuning?* No. Tuning and selection use only the training
  split; the test split is used once.
- *Why not nested cross-validation?* It would give an unbiased estimate of the whole selection
  procedure, at 5× the cost. A single held-out split gives an unbiased estimate of the chosen model,
  which is what the user needs, and is cheaper.
- *What if the dataset is tiny?* Fold count drops to the size of the smallest class; below 100 rows
  the UI warns that metrics are unreliable.

## 2. Preprocessing

### Preprocessing lives inside the scikit-learn pipeline
`build_preprocessor`: imputation, scaling and encoding are steps of the same `Pipeline` as the model.

**Why:** if you scale or impute using the whole dataset before splitting, statistics from the
validation rows leak into training. Inside the pipeline, each CV fold fits its own imputer and
scaler on its training rows only. It also means the saved model carries its preprocessing, so
prediction on new data applies exactly the same transformations.

### Specific choices
- **Median imputation** for numbers (robust to outliers), **most frequent** for categories.
- **One-hot encoding capped at 30 categories**, rare ones grouped, unknown categories at prediction
  time tolerated. Uncapped one-hot on a high-cardinality column creates thousands of sparse columns.
- **Dates** become year, month, day of week and days since the earliest training date
  (`DatetimeFeatures`). One-hot encoding dates gives one column per date, which cannot generalise.
- **ID-like columns are dropped**: non-numeric columns where more than 95% of values are unique.
  They carry no pattern and invite memorisation.
- **Class weights** are balanced when the smallest class is less than half the size of the largest.

### Task type detection
`detect_task_type`: text or boolean targets are classification; numeric targets are classification
if they have ≤ 5 distinct values, or are integer-valued with ≤ 20 distinct values covering ≤ 5% of
rows (digit labels, star ratings); otherwise regression. The user can override it.

**Why a rule plus an override:** no rule is right for every dataset, so the rule handles the common
cases and the override handles the rest.

**Likely questions**
- *Why not target encoding for high-cardinality categories?* It needs careful out-of-fold handling
  to avoid leakage; capped one-hot is safe by construction.
- *Why StandardScaler for tree models, which do not need it?* One preprocessing path for every
  algorithm keeps the comparison fair and the code simple; scaling does not hurt trees.

## 3. Explainability

### Permutation importance, on the test split, per original column
`compute_feature_importance`: shuffle one input column, measure how much the score drops, repeat 5
times, report mean and standard deviation.

**Why not the tree models' built-in importance:** it only exists for trees, it is computed on
training data, and it is biased towards high-cardinality features. Permutation importance works for
every algorithm, so SVM and logistic regression get explanations too, and it measures effect on
held-out performance.
**Why not SHAP:** SHAP gives per-prediction explanations, which this does not. It is a heavy
dependency and slow for kernel models. It is the natural next step.
**Known weakness:** when two features are strongly correlated, shuffling one barely hurts because
the other still carries the information, so both look less important than they are.

### ROC AUC without probabilities
`classification_diagnostics` uses `predict_proba` where it exists and `decision_function` otherwise
(SVM), and computes one-vs-rest AUC per class, macro-averaged for multiclass.

## 4. Leakage detection

`backend/services/leakage_detector.py` flags columns that give the answer away:

1. **Single-feature predictive power**: a depth-5 tree is cross-validated on each feature alone. A
   feature that scores ≥ 0.97 by itself is flagged. This catches categorical and non-linear leaks.
2. **Exact copies** of the target.
3. **Correlation ≥ 0.95**, for numeric and binary targets only. Correlation with integer class
   codes is meaningless for three or more classes, so it is skipped there.
4. **Arithmetic relationships** (`a + b`, `a - b`, `a × b` equals the target), checked among the
   nine columns most correlated with the target, because checking every pair is quadratic.
5. **Names**, matched by whole word, so `customer_id` is flagged and `paid` is not.

**The honest caveat:** these are heuristics. A flag means "check whether this value is known before
the outcome happens", not proof. A legitimately strong feature can trip check 1, and a subtle
temporal leak can pass all five.

## 5. Platform

### Background jobs in the database, not Celery
`backend/services/jobs.py`. Training returns `202` with a job id; the UI polls for progress.

**How a job is never run twice:** a worker claims a job with a single conditional
`UPDATE ... WHERE status = 'queued'`. The database guarantees only one worker's update matches.
**Crash recovery:** a running job heartbeats every 15 s; one silent for 120 s is requeued, up to 3
attempts.
**Why not Celery or RQ:** they need Redis or RabbitMQ. The database already holds job status for the
UI, so using it as the queue adds no infrastructure and works on Windows. The cost is throughput,
which does not matter for training jobs that take seconds to minutes.

### Storage
- **Content-addressed uploads**: a file is stored under its SHA-256 hash, so the same data is never
  stored twice and a user-supplied filename is never used as a path (no path traversal).
- **Three interchangeable backends** behind one interface: local disk, S3, lakeFS.
- **Models are pickles, and unpickling runs code**, so a model is only ever loaded if it is
  registered to the requesting user and its storage key was generated by the server
  (`load_model_artifact`).

### Authentication
Argon2 password hashing, JWT access tokens, and a router-level dependency so a new endpoint cannot
be left public by accident (a test enumerates every route and checks). Every query filters by
`owner_id`. In local development `AUTH_REQUIRED` is off and requests act as one built-in user;
production refuses to start with it off.

## 6. MLOps integrations

### MLflow is an optional mirror, not the source of truth
`backend/services/mlflow_tracking.py`. StatMind's own tables drive the app; when
`MLFLOW_TRACKING_URI` is set, each model is also logged to MLflow.

**Why keep both:** the app must work with no extra services, and must not fail a training job
because a tracking server is down. Every MLflow call is wrapped, so a failure is logged and ignored.
**AutoML** logs the winner as a parent run and each candidate as a child run.
**Stages:** promoting a model in StatMind moves MLflow's `production` alias, so
`models:/statmind-<target>@production` always resolves to the current production model.

### lakeFS versions the files
`backend/services/lakefs_versioning.py`. lakeFS speaks the S3 protocol, so the existing S3 storage
code works against it: the bucket is the repository and the first path segment is the branch.

**What is added:** a commit after each dataset upload and each trained model. A model records the
commit holding its training data and the commit holding the model file, which answers "exactly
which data produced this model?" in a way that can be checked: the file can be read back at that commit.
**Why lakeFS and not DVC:** DVC versions files alongside a Git repository and suits a developer's
workflow; lakeFS versions a running application's object storage through an API, which is what
this is.
**Failure policy:** commits are best effort, startup is strict. A failed commit must not lose an
upload; an unreachable repository means nothing can be stored at all.

### The AI analyst is grounded and has a fallback
`backend/services/ai_analyst.py` sends the model a JSON snapshot of what StatMind computed and tells
it to answer only from that. With no API key, or on any API error, keyword rules answer instead,
and the UI says which one replied. Tests never touch the network.

## 7. Serving and monitoring

### The production stage decides which model answers
`backend/services/serving.py`. `POST /api/predict?target=churned` uses whichever model is in
production for that target, and promoting a model retires the previous one.

**Why:** callers should not hold model ids. Rolling out a new model, or rolling back, is then a
change in the registry and nothing else. **One production model per target** removes any ambiguity
about which one answers.

**Why serve inside the API instead of a model server (BentoML, KServe, MLflow serving):** one fewer
service to run, and it is enough for a few requests a second. The trade-off is no independent
scaling and no batching; at real traffic the model server is the right answer.

**Why cache models in memory:** unpickling a pipeline per request costs more than the prediction.
Saved models are immutable, so the cache is always correct; it is bounded to eight models.

### Drift is tested against the training data, per column
`backend/services/drift.py`, using Evidently.

**Why drift at all:** a model is validated once, on a split of old data. Nothing in that number
tells you when the world has moved. Comparing incoming data with training data is the earliest
signal available, because it needs no labels.

**Why not measure accuracy instead:** that needs the true outcome for each prediction, which
arrives later or never (did the customer churn?). It is the better measurement and is not built.

**Why correct for multiple tests:** each column is tested at the 0.05 level. With 20 columns that
never changed, the chance at least one is flagged is 1 − 0.95²⁰ ≈ 64%. Dividing the threshold by
the number of columns (Bonferroni) keeps the chance of any false alarm near 5%, at the cost of
missing small shifts. For large samples Evidently switches from p-values to distances
(Wasserstein, Jensen-Shannon), because with thousands of rows every tiny difference is
"significant".

**Likely questions**
- *A column drifted. Is the model broken?* Not necessarily. If the column barely matters to the
  model, the drift is harmless. Cross-check it with the permutation importance.
- *What would you do on a drift alarm?* Look at which columns moved and whether that is expected
  (a price change, a new region), check recent outcomes if any exist, then retrain on newer data.
- *Data drift versus concept drift?* Data drift is the inputs changing, which this detects.
  Concept drift is the relationship between inputs and outcome changing; it is invisible without
  labels.

## 8. Known limitations

- Everything is in-memory pandas; uploads are capped at 50 MB.
- AutoML cross-validates on at most 20,000 rows and skips SVM and KNN above 5,000.
- Predictions are served from inside the API process; there is no separate model server.
- Drift is checked on request, not on a schedule; true outcomes are not recorded, so accuracy over
  time is not measured; nothing retrains automatically.
- Saved models are pickles tied to this codebase and its library versions.
- lakeFS uses one branch; prediction files and reports are committed only with the next dataset or model.
- The benchmark results are on small textbook datasets.
