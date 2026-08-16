# Data Source Register

This register records datasets approved for future investigation. Approval for investigation is not permission to download or use a dataset. Source authenticity, current licensing terms, access conditions, attribution requirements, and intended-use compatibility must be verified and documented before acquisition.

## Approved future datasets

### Olist

- **Intended capability:** fulfillment and delivery intelligence
- **Planned role:** support research into operational and delivery patterns that may inform fulfillment decisions
- **Pre-download requirement:** verify the authoritative source, version, license or usage terms, access conditions, and permitted project use

No schema, column availability, or analytical outcome is assumed at this stage.

### Instacart

- **Intended capability:** personalization and recommendation
- **Planned role:** support research into product discovery and recommendation behavior
- **Pre-download requirement:** verify the authoritative source, version, license or usage terms, access conditions, and permitted project use

No schema, column availability, or analytical outcome is assumed at this stage.

### Amazon Reviews 2023 — Appliances only

- **Intended capability:** voice of customer
- **Planned role:** support research into review sentiment, recurring concerns, and customer-reported issues within the Appliances category only
- **Pre-download requirement:** verify the authoritative source, category scope, version, license or usage terms, access conditions, and permitted project use

No schema, column availability, or analytical outcome is assumed at this stage.

## Dataset separation policy

- **Do not merge these datasets.**
- Each dataset serves an independent AI capability and must retain its own provenance, processing, evaluation, and limitations.
- Records across datasets must not be interpreted as representing the same customers, products, orders, sellers, or events.
- The capabilities will later connect only through a shared **Decision Action Card API contract** that standardizes decision outputs, not source data.
- Licensing and source details must be verified and recorded before any dataset is downloaded.

