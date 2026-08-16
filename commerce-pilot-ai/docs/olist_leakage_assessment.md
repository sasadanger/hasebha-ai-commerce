# Olist Leakage Assessment

## Observed event fields

The orders table contains purchase, approval, carrier-handoff, customer-delivery, and estimated-delivery timestamps plus final status. Items contain a shipping-limit timestamp. Reviews contain creation and answer timestamps. These fields occur at different points in an order lifecycle.

## Prediction-point assessment

| Field group | At order creation | At payment approval | At shipment handoff |
|---|---|---|---|
| Purchase-time facts and customer/order identifiers | Potentially available | Available | Available |
| Approval timestamp and payment records | Forbidden/not yet known | Available only after approval is confirmed | Available |
| Item/seller facts and precommitted shipping limit | Availability must be verified | Potentially available | Available if recorded before handoff |
| Carrier-handoff timestamp | Forbidden | Forbidden | Available at confirmed handoff |
| Customer-delivery timestamp | Forbidden | Forbidden | Forbidden |
| Final status, later cancellation state, reviews | Forbidden | Forbidden | Forbidden |
| Estimated-delivery date | Allowed only if it was the estimate visible at that prediction time | Same condition | Same condition |

Availability is a production-system fact, not proven merely because a field exists in this historical export. The observed 1,359 carrier-before-approval and 23 delivery-before-carrier sequences reinforce the need for explicit event-quality rules.

## Conclusion

The safest initial research prediction point is **confirmed payment approval**: it is early enough for operational intervention and offers a clearer information boundary than creation, while avoiding carrier and delivery outcomes. Shipment handoff is a valid later alternative for carrier-stage decisions but answers a different business question.

Actual delivery timestamp, final status, review data, and any later event are leakage for both options. No feature matrix was created.

