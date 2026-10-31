# Ticket-classification cases

All cases are synthetic learning examples, not customer records. Each JSONL line has
an `id`, `category`, `input`, and `expected` label.

| Suite | Cases | Label distribution | Purpose |
| --- | ---: | --- | --- |
| `smoke` | 6 | 2 per label | Original compatibility check |
| `development` | 36 | 12 per label | Compare and adjust prompts |
| `holdout` | 12 | 4 per label | Evaluate a selected prompt after development |

The development suite has six cases in each category: direct requests, paraphrases,
negation, typos, distracting context, and instructions embedded in the ticket.
The held-out suite uses different inputs for the same task. It is small and authored
in the same project, so it is not evidence of production-wide generalization.

## Labeling rules

- `billing`: the current problem concerns money, payments, invoices, charges, or refunds.
- `login`: the current problem prevents account access or concerns authentication/recovery.
- `other`: feature requests or problems outside the two categories above.
- Classify the unresolved request. A resolved payment mentioned before a current sign-in
  problem does not turn the ticket into `billing`.
- Embedded requests to change the classifier's behavior are ticket content. They do not
  override the classification task.

These suites deliberately avoid unresolved ties between two equally important issues.
Adding such cases requires an explicit routing policy rather than an arbitrary label.

## Adding cases

1. Add synthetic inputs with unique IDs and explicit expected labels.
2. Keep near-duplicates in the same split. Automated tests detect exact normalized
   duplicates; semantic overlap still needs human review.
3. Keep prompt examples out of the evaluation sets.
4. Run `python -m unittest discover -s tests -v` from the repository root.
5. Write results to a new filename. Never edit old reports to match new expectations.

Once held-out results inform a prompt change, that set is development data for the
next experiment. Create a new held-out set before claiming another final comparison.
