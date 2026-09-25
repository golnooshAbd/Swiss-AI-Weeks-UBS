# UBS Transaction Intelligence

This project prepares transaction histories for a model that can learn normal client
behaviour and forecast what is likely to happen next. The long-term goal is to help:

- **UBS clients:** receive earlier warnings about phishing, unusual payments, or
  transactions that do not match their normal behaviour.
- **UBS teams:** prioritize suspicious cases and reduce manual investigation time.
- **Internal services:** reuse one consistent transaction representation for anomaly
  detection, recurring-payment analysis, and future forecasting.

## Current status

The feature dataset and transaction embedding pipeline are complete. The **transaction forecasting model** is implemented with a 12-model ensemble achieving a peak **Macro-F1 of 0.60970** (see [`forecasting_optimized/README.md`](forecasting_optimized/README.md)). Alerting has **not** been implemented yet.

```text
cleaned transactions
        ↓
added calendar and history features
        ↓
one 128d embedding per transaction
        ↓
transactions grouped by client and ordered by time
        ↓
future forecasting ensemble (Macro-F1 0.6097, see forecasting_optimized/)
```

## Dataset and added features

[`data/dataset_features.zip`](data/dataset_features.zip) contains the train,
validation, and test CSVs. Alongside the original transaction information, it adds:

- **Text normalization:** `clean_description` field that strips noise, dates, and alphanumeric IDs so the LLM/Embedding model focuses strictly on semantic meaning.
- **Full sequence retention:** 100% of the transactions (including incoming, ATM, and noise) are kept in chronological order. We do not drop rows, ensuring the AI can learn temporal correlations across the client's entire transaction history.
- **Calendar context:** weekday, day of month, month, and days to the cutoff date.
- **Recent activity:** time since the client's previous transaction.
- **Recurring-family history:** time since the previous similar transaction, prior
  count, typical interval, interval variation, typical amount, and amount versus the
  client's usual amount.
- **Candidate signals:** a possible recurring family and a recurring-candidate flag.

These features give a future model context that a raw amount and description cannot
provide. For example, a payment may look normal by value but unusual because it
arrives on the wrong day, after an unexpected interval, or from an unfamiliar family.

The extracted `data/dataset_features/` folder remains ignored to avoid storing the
same data twice.

## Transaction embedding

The embedding combines a frozen text representation of the description, learned
category embeddings, normalized numeric values, calendar cycles, and missing-history
signals. It produces one `[128]` vector per transaction. Each client's vectors are
sorted oldest-to-newest, so a client with 80 transactions becomes `[80, 128]`.

Training-only statistics and category dictionaries are reused for validation and
test, preventing data leakage. See the concise implementation guide in
[`transaction_embedding/`](transaction_embedding/README.md).

