/**
 * Admin extension: surfaces the HASEBHA Intelligence data (internal codename:
 * CommercePilot) persisted on an order's metadata (by
 * apps/backend/src/subscribers/order-placed.ts) directly on the order
 * detail page. Reads only `order.metadata.commercepilot_ai` -- this metadata
 * key is an integration contract and is intentionally left unchanged by the
 * HASEBHA brand rename. No new API route, no new module, no separate data
 * fetch: the order detail page already loads the order (metadata included),
 * and this widget is handed that same data via Medusa's
 * DetailWidgetProps<AdminOrder>.
 */
import { defineWidgetConfig } from "@medusajs/admin-sdk"
import { Badge, Container, Heading, StatusBadge, Text } from "@medusajs/ui"
import type { AdminOrder, DetailWidgetProps } from "@medusajs/types"

type ProcessingStatus =
  | "COMPLETED"
  | "AI_UNAVAILABLE"
  | "AI_TIMEOUT"
  | "AI_REJECTED_INVALID_FEATURES"
  | "INSUFFICIENT_FEATURES"

type CommercePilotAiRecord = {
  idempotency_key: string
  processing_status: ProcessingStatus
  order_id: string
  fulfillment_risk?: {
    risk_score: number
    risk_class: string
    model_version: string
    model_experiment_id: string
    model_artifact_sha256: string
  }
  decision?: {
    action: string
    priority: string
    reason_codes: string[]
    ruleset_version: string
  }
  feature_source?: {
    purchase_timestamp: string
    approval_timestamp: string
    approval_source: string
  }
  error?: string
  processed_at: string
}

const STATUS_COLOR: Record<ProcessingStatus, "green" | "red" | "orange" | "grey"> = {
  COMPLETED: "green",
  AI_UNAVAILABLE: "orange",
  AI_TIMEOUT: "orange",
  AI_REJECTED_INVALID_FEATURES: "red",
  INSUFFICIENT_FEATURES: "grey",
}

const PRIORITY_COLOR: Record<string, "green" | "red" | "orange" | "blue" | "grey"> = {
  P0_CRITICAL: "red",
  P1_HIGH: "red",
  P2_SUPPRESS_MARKETING: "orange",
  P3_GROWTH: "blue",
  P4_ROUTINE: "grey",
}

const CommercePilotAiOrderWidget = ({ data: order }: DetailWidgetProps<AdminOrder>) => {
  const record = (order.metadata?.commercepilot_ai ?? null) as CommercePilotAiRecord | null

  if (!record) {
    return (
      <Container className="divide-y p-0">
        <div className="flex items-center justify-between px-6 py-4">
          <Heading level="h2">HASEBHA Intelligence</Heading>
        </div>
        <div className="px-6 py-4">
          <Text size="small" className="text-ui-fg-subtle">
            No HASEBHA Intelligence record yet for this order. The order.placed subscriber
            processes new orders asynchronously; refresh in a moment.
          </Text>
        </div>
      </Container>
    )
  }

  return (
    <Container className="divide-y p-0">
      <div className="flex items-center justify-between px-6 py-4">
        <Heading level="h2">HASEBHA Intelligence</Heading>
        <StatusBadge color={STATUS_COLOR[record.processing_status] ?? "grey"}>
          {record.processing_status}
        </StatusBadge>
      </div>

      {record.fulfillment_risk && (
        <div className="px-6 py-4">
          <Text size="small" weight="plus" className="mb-2">
            AI Risk Signal
          </Text>
          <div className="flex items-center gap-x-2">
            <Text size="small" className="text-ui-fg-subtle">
              risk_score
            </Text>
            <Text size="small" weight="plus">
              {record.fulfillment_risk.risk_score.toFixed(4)}
            </Text>
          </div>
          <div className="flex items-center gap-x-2 mt-1">
            <Text size="small" className="text-ui-fg-subtle">
              business-rule classification
            </Text>
            <Badge color={record.fulfillment_risk.risk_class === "high" ? "red" : "green"}>
              {record.fulfillment_risk.risk_class}
            </Badge>
          </div>
          <Text size="small" className="text-ui-fg-subtle mt-2">
            risk_score is the model&apos;s raw output, not a calibrated
            probability. The risk_class threshold has a documented
            discrepancy under review; treat risk_score as the primary
            signal and risk_class as an approximate, rule-based label.
          </Text>
        </div>
      )}

      {record.decision && (
        <div className="px-6 py-4">
          <Text size="small" weight="plus" className="mb-2">
            Decision Engine
          </Text>
          <div className="flex items-center gap-x-2">
            <Badge color={PRIORITY_COLOR[record.decision.priority] ?? "grey"}>{record.decision.priority}</Badge>
            <Text size="small">{record.decision.action}</Text>
          </div>
          <Text size="small" className="text-ui-fg-subtle mt-1">
            reason codes: {record.decision.reason_codes.join(", ")} &middot; ruleset {record.decision.ruleset_version}
          </Text>
        </div>
      )}

      {record.feature_source && (
        <div className="px-6 py-4">
          <Text size="small" weight="plus" className="mb-2">
            Feature source
          </Text>
          <Text size="small" className="text-ui-fg-subtle">
            approval_timestamp derived from {record.feature_source.approval_source}
          </Text>
        </div>
      )}

      {record.error && (
        <div className="px-6 py-4">
          <Text size="small" weight="plus" className="mb-2">
            Error
          </Text>
          <Text size="small" className="text-ui-fg-subtle">
            {record.error}
          </Text>
        </div>
      )}

      <div className="px-6 py-4">
        <Text size="small" weight="plus" className="mb-2">
          Model / Audit
        </Text>
        {record.fulfillment_risk && (
          <Text size="small" className="text-ui-fg-subtle">
            model {record.fulfillment_risk.model_experiment_id} (v{record.fulfillment_risk.model_version}, artifact {record.fulfillment_risk.model_artifact_sha256.slice(0, 12)})
          </Text>
        )}
        <Text size="small" className="text-ui-fg-subtle">
          status {record.processing_status} &middot; processed {record.processed_at}
        </Text>
        <Text size="small" className="text-ui-fg-subtle">
          idempotency key {record.idempotency_key}
        </Text>
      </div>
    </Container>
  )
}

export const config = defineWidgetConfig({
  zone: "order.details.side.after",
})

export default CommercePilotAiOrderWidget
