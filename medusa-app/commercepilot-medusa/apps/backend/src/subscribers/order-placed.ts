/**
 * CommercePilot AI integration -- calls the FastAPI ai_service on order
 * placement for fulfillment-risk scoring and a decision-engine
 * recommendation, then persists an auditable result onto the order itself.
 *
 * Design boundaries (see reports/checkpoints/commercepilot_live_e2e_gate_2026-08-15/
 * in the commerce-pilot-ai repo for the full investigation this is based on):
 *
 * 1. FEATURE CONTRACT: the frozen Olist model was trained on
 *    `purchase_timestamp`/`approval_timestamp` -- real Brazilian-marketplace
 *    payment-approval timing. Medusa has no field literally named
 *    "approval", so this subscriber derives approval_timestamp only from
 *    REAL, already-recorded events on the order's payment record(s)
 *    (`payment.captured_at`, falling back to `payment.created_at` --
 *    verified via Medusa's own DML schema, not guessed): never from
 *    "now"/event-processing time, which would be a fabricated stand-in, not
 *    a derived feature. If no payment record exists yet at event time, this
 *    explicitly records INSUFFICIENT_FEATURES rather than inventing a value
 *    -- see resolveOlistFeatures().
 * 2. PERSISTENCE: the AI result is written into the order's own `metadata`
 *    field via the Order module service's supported `updateOrders` method
 *    -- no direct DB write, no new module/migration/infrastructure.
 * 3. IDEMPOTENCY: `order.placed` fires once per order in normal operation,
 *    so the order's own id is the natural deterministic idempotency key.
 *    Before doing anything, this checks whether `metadata.commercepilot_ai`
 *    already carries that key and returns early if so -- redelivery is a
 *    safe no-op, not a duplicate decision. (Known limitation, documented,
 *    not engineered around: two *concurrent* deliveries of the same event
 *    landing before either has persisted yet could both proceed -- a
 *    DB-level unique constraint/lock would close this but is out of scope
 *    for a single-instance dev integration proof.)
 * 4. FAIL-SOFT + BOUNDED RETRIES: every AI call has a bounded timeout and a
 *    small number of retries with backoff; order processing is never
 *    blocked, and every outcome (success, unavailable, timeout, rejected,
 *    insufficient features) is persisted as an explicit processing_status,
 *    not just logged.
 */
import type { SubscriberArgs, SubscriberConfig } from "@medusajs/framework"
import { ContainerRegistrationKeys, Modules } from "@medusajs/framework/utils"

const AI_SERVICE_URL = process.env.COMMERCEPILOT_AI_SERVICE_URL || "http://localhost:8123"
const AI_SERVICE_API_KEY = process.env.COMMERCEPILOT_AI_SERVICE_API_KEY || ""
const REQUEST_TIMEOUT_MS = 5000
const MAX_RETRIES = 2
const RETRY_BACKOFF_MS = [200, 500]

type FulfillmentRiskResponse = {
  risk_score: number
  risk_class: "low" | "high"
  model_version: string
  model_experiment_id: string
  model_artifact_sha256: string
}

type DecisionResponse = {
  action: string
  priority: string
  reason_codes: string[]
  ruleset_version: string
}

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
    approval_source: "payment.captured_at" | "payment.created_at"
  }
  error?: string
  processed_at: string
}

type AiCallResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; timedOut: boolean; rejected: boolean }

async function postJsonWithRetry<T>(path: string, body: unknown): Promise<AiCallResult<T>> {
  let lastError = ""
  let timedOut = false
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${AI_SERVICE_URL}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(AI_SERVICE_API_KEY ? { "X-Internal-Api-Key": AI_SERVICE_API_KEY } : {}),
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      })
      if (res.status === 422) {
        // Explicit, deterministic rejection (e.g. invalid/inverted
        // timestamps) -- not transient, so do not retry.
        const text = await res.text()
        return { ok: false, error: `422: ${text}`, timedOut: false, rejected: true }
      }
      if (res.ok) {
        return { ok: true, data: (await res.json()) as T }
      }
      lastError = `${res.status}: ${await res.text()}`
    } catch (err: unknown) {
      const name = (err as { name?: string } | undefined)?.name
      timedOut = name === "TimeoutError" || name === "AbortError"
      lastError = String(err)
    }
    if (attempt < MAX_RETRIES) {
      await new Promise((resolve) => setTimeout(resolve, RETRY_BACKOFF_MS[attempt] ?? 500))
    }
  }
  return { ok: false, error: lastError, timedOut, rejected: false }
}

type FeatureResolution =
  | {
      status: "AVAILABLE_NOW"
      purchaseTimestamp: string
      approvalTimestamp: string
      approvalSource: "payment.captured_at" | "payment.created_at"
    }
  | { status: "UNAVAILABLE_AT_EVENT_TIME"; reason: string }

/**
 * AVAILABLE_NOW vs UNAVAILABLE_AT_EVENT_TIME adapter boundary. Uses
 * Medusa's Query API (the supported way to read cross-module-linked data --
 * payment_collections/payments belong to the Payment module, linked to
 * Order via a remote link, not a direct relation) to read only fields that
 * represent real, already-happened events.
 */
async function resolveOlistFeatures(
  container: SubscriberArgs<unknown>["container"],
  orderId: string
): Promise<FeatureResolution> {
  const query = container.resolve(ContainerRegistrationKeys.QUERY)
  const { data } = await query.graph({
    entity: "order",
    filters: { id: orderId },
    fields: [
      "id",
      "created_at",
      "payment_collections.payments.id",
      "payment_collections.payments.captured_at",
      "payment_collections.payments.canceled_at",
      "payment_collections.payments.created_at",
    ],
  })
  const order = data[0] as
    | {
        created_at?: string
        payment_collections?: Array<{
          payments?: Array<{ captured_at?: string; canceled_at?: string; created_at?: string }>
        }>
      }
    | undefined

  if (!order || !order.created_at) {
    return { status: "UNAVAILABLE_AT_EVENT_TIME", reason: "order or order.created_at not found via query.graph" }
  }
  const purchaseTimestamp = new Date(order.created_at).toISOString()

  const payments = (order.payment_collections ?? []).flatMap((pc) => pc.payments ?? [])
  const active = payments.filter((p) => !p.canceled_at)

  const captured = active
    .filter((p): p is { captured_at: string; canceled_at?: string; created_at?: string } => !!p.captured_at)
    .sort((a, b) => new Date(a.captured_at).getTime() - new Date(b.captured_at).getTime())[0]
  if (captured) {
    return {
      status: "AVAILABLE_NOW",
      purchaseTimestamp,
      approvalTimestamp: new Date(captured.captured_at).toISOString(),
      approvalSource: "payment.captured_at",
    }
  }

  // captured_at is frequently null until a separate capture step/webhook
  // runs (provider-dependent); payment.created_at is the moment Medusa
  // recorded the authorized payment -- a real event, not "now".
  const earliestPayment = active
    .filter((p): p is { created_at: string; captured_at?: string; canceled_at?: string } => !!p.created_at)
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())[0]
  if (earliestPayment) {
    return {
      status: "AVAILABLE_NOW",
      purchaseTimestamp,
      approvalTimestamp: new Date(earliestPayment.created_at).toISOString(),
      approvalSource: "payment.created_at",
    }
  }

  return {
    status: "UNAVAILABLE_AT_EVENT_TIME",
    reason:
      "no non-canceled payment record found on order at order.placed time -- approval_timestamp cannot be " +
      "legitimately derived without fabricating a value",
  }
}

function statusFromAiFailure(result: { rejected: boolean; timedOut: boolean }): ProcessingStatus {
  if (result.rejected) return "AI_REJECTED_INVALID_FEATURES"
  if (result.timedOut) return "AI_TIMEOUT"
  return "AI_UNAVAILABLE"
}

/**
 * SHADOW-MODE production-parity Seller-SLA feature resolution.
 *
 * Distinct from resolveOlistFeatures() above: that one resolves the 2
 * timing fields the FROZEN V1 model needs. This resolves the 9 real,
 * genuinely-available fields the NEW production-parity model needs (see
 * commerce-pilot-ai/reports/generated/olist_v3_multistage/
 * SELLER_SLA_SINGLE_VENDOR_PARITY_REAUDIT.json for the audit this follows).
 *
 * `same_zone` is resolved via resolveSingleStoreShipFromProvince() -- a real
 * comparison between the customer's shipping province and the store's
 * ship-from province, valid when the store has exactly one configured stock
 * location (the expected single-vendor case). It is never asserted `true`
 * without both real provinces being known and matching; fabricating this
 * boolean would violate the project's "no fake feature-parity mapping" rule.
 */
type ProductionParityFeatureResolution =
  | {
      status: "AVAILABLE_NOW"
      purchaseTimestamp: string
      nItems: number
      nDistinctProducts: number
      nCategories: number
      totalPrice: number
      totalFreight: number
      weightG: number
      volumeCm3: number
      paymentValue: number
      sameZone: boolean
    }
  | { status: "UNAVAILABLE_AT_EVENT_TIME"; reason: string }

/**
 * Resolves the store's ship-from province at T0 (order-placement time),
 * WITHOUT relying on order.fulfillments -- a fulfillment record does not
 * exist yet at order.placed time (it is created later, at pack/ship), so
 * `fulfillments.location_id` is never populated this early. Fabricating a
 * ship-from guess from a not-yet-existent fulfillment would violate this
 * project's "no fake feature values" rule.
 *
 * Instead this queries the StockLocation module directly (a T0-available,
 * real store-configuration fact, not an order-derived one). If the store has
 * exactly ONE stock location -- the expected, common case for a single-vendor
 * deployment like HASEBHA -- its address province is the legitimate ship-from
 * zone for every order. If the store has zero or more than one location, this
 * honestly returns null (UNAVAILABLE) rather than guessing which location
 * will eventually fulfill this specific order.
 */
async function resolveSingleStoreShipFromProvince(
  container: SubscriberArgs<unknown>["container"]
): Promise<string | null> {
  const query = container.resolve(ContainerRegistrationKeys.QUERY)
  const { data: locations } = await query.graph({
    entity: "stock_location",
    fields: ["id", "address.province"],
  })
  if (!locations || locations.length !== 1) {
    return null
  }
  const province = (locations[0] as { address?: { province?: string | null } }).address?.province
  return province ?? null
}

async function resolveProductionParityFeatures(
  container: SubscriberArgs<unknown>["container"],
  orderId: string
): Promise<ProductionParityFeatureResolution> {
  const query = container.resolve(ContainerRegistrationKeys.QUERY)
  const { data } = await query.graph({
    entity: "order",
    filters: { id: orderId },
    fields: [
      "id",
      "created_at",
      "item_total",
      "shipping_total",
      "shipping_address.province",
      "items.quantity",
      "items.product_id",
      "items.product.weight",
      "items.product.length",
      "items.product.height",
      "items.product.width",
      "items.product.categories.id",
      "payment_collections.payments.amount",
      "payment_collections.payments.canceled_at",
    ],
  })
  const order = data[0] as
    | {
        created_at?: string
        item_total?: number
        shipping_total?: number
        shipping_address?: { province?: string }
        items?: Array<{
          quantity?: number
          product_id?: string
          product?: {
            weight?: number | null
            length?: number | null
            height?: number | null
            width?: number | null
            categories?: Array<{ id?: string }>
          }
        }>
        payment_collections?: Array<{ payments?: Array<{ amount?: number; canceled_at?: string }> }>
      }
    | undefined

  if (!order || !order.created_at) {
    return { status: "UNAVAILABLE_AT_EVENT_TIME", reason: "order or order.created_at not found via query.graph" }
  }
  const items = order.items ?? []
  if (items.length === 0) {
    return { status: "UNAVAILABLE_AT_EVENT_TIME", reason: "order has no items at order.placed time" }
  }

  const nItems = items.reduce((sum, it) => sum + (it.quantity ?? 0), 0) || items.length
  const distinctProductIds = new Set(items.map((it) => it.product_id).filter(Boolean))
  const distinctCategoryIds = new Set(
    items.flatMap((it) => (it.product?.categories ?? []).map((c) => c.id).filter(Boolean))
  )

  // Weight/volume: real product dimension fields, summed across line items.
  // A null dimension on a given product is a real, honest data-quality gap
  // (not every catalog populates these) -- recorded as 0 for that line, not
  // fabricated as a nonzero guess, consistent with the parity re-audit.
  let weightG = 0
  let volumeCm3 = 0
  for (const it of items) {
    const qty = it.quantity ?? 1
    const w = it.product?.weight ?? 0
    const l = it.product?.length ?? 0
    const h = it.product?.height ?? 0
    const wd = it.product?.width ?? 0
    weightG += w * qty
    volumeCm3 += l * h * wd * qty
  }

  const payments = (order.payment_collections ?? []).flatMap((pc) => pc.payments ?? [])
  const activePayments = payments.filter((p) => !p.canceled_at)
  const paymentValue = activePayments.reduce((sum, p) => sum + (p.amount ?? 0), 0)

  // same_zone: real comparison between the customer's shipping province and
  // the store's ship-from province (resolveSingleStoreShipFromProvince,
  // T0-available -- see that function's own docstring for why it does NOT
  // use order.fulfillments, which do not exist yet at order.placed time).
  // If the ship-from province cannot be determined (zero or multiple stock
  // locations) or the customer province is missing, this is recorded as
  // `false` -- the same conservative default as before -- but the REASON is
  // now an honestly-resolved "unknown, so not asserted true" rather than a
  // permanently hardcoded stub. Never asserted `true` without both real
  // provinces being known and matching.
  const storeProvince = await resolveSingleStoreShipFromProvince(container)
  const customerProvince = order.shipping_address?.province ?? null
  const sameZone = storeProvince !== null && customerProvince !== null && storeProvince === customerProvince

  return {
    status: "AVAILABLE_NOW",
    purchaseTimestamp: new Date(order.created_at).toISOString(),
    nItems,
    nDistinctProducts: Math.max(distinctProductIds.size, 1),
    nCategories: Math.max(distinctCategoryIds.size, 1),
    totalPrice: order.item_total ?? 0,
    totalFreight: order.shipping_total ?? 0,
    weightG,
    volumeCm3,
    paymentValue,
    sameZone,
  }
}

export default async function orderPlacedAiHandler({
  event: { data },
  container,
}: SubscriberArgs<{ id: string }>) {
  const logger = container.resolve(ContainerRegistrationKeys.LOGGER)
  const orderModuleService = container.resolve(Modules.ORDER)
  const idempotencyKey = `order.placed:${data.id}`

  let order
  try {
    order = await orderModuleService.retrieveOrder(data.id)
  } catch (err) {
    logger.warn(`[commercepilot-ai] could not retrieve order ${data.id}: ${err}`)
    return
  }

  const existingMetadata = (order.metadata ?? {}) as Record<string, unknown>
  const existingRecord = existingMetadata.commercepilot_ai as CommercePilotAiRecord | undefined
  if (existingRecord && existingRecord.idempotency_key === idempotencyKey) {
    logger.info(
      `[commercepilot-ai] order ${data.id} already processed (status=${existingRecord.processing_status}); ` +
        `skipping duplicate order.placed delivery (idempotency_key=${idempotencyKey})`
    )
    return
  }

  const persist = async (record: CommercePilotAiRecord) => {
    await orderModuleService.updateOrders(data.id, {
      metadata: { ...existingMetadata, commercepilot_ai: record },
    })
  }

  const features = await resolveOlistFeatures(container, data.id)
  if (features.status === "UNAVAILABLE_AT_EVENT_TIME") {
    logger.warn(
      `[commercepilot-ai] order ${data.id}: insufficient features (${features.reason}); recording ` +
        `INSUFFICIENT_FEATURES rather than fabricating data`
    )
    await persist({
      idempotency_key: idempotencyKey,
      processing_status: "INSUFFICIENT_FEATURES",
      order_id: data.id,
      error: features.reason,
      processed_at: new Date().toISOString(),
    })
    return
  }

  const featureSource = {
    purchase_timestamp: features.purchaseTimestamp,
    approval_timestamp: features.approvalTimestamp,
    approval_source: features.approvalSource,
  }

  const riskResult = await postJsonWithRetry<FulfillmentRiskResponse>("/v1/fulfillment/risk", {
    order_ref: data.id,
    purchase_timestamp: features.purchaseTimestamp,
    approval_timestamp: features.approvalTimestamp,
  })

  if (!riskResult.ok) {
    const status = statusFromAiFailure(riskResult)
    logger.warn(
      `[commercepilot-ai] order ${data.id}: fulfillment risk call failed (${status}): ${riskResult.error}. ` +
        `Order processing continues unaffected -- this is an advisory signal, not a checkout gate.`
    )
    await persist({
      idempotency_key: idempotencyKey,
      processing_status: status,
      order_id: data.id,
      feature_source: featureSource,
      error: riskResult.error,
      processed_at: new Date().toISOString(),
    })
    return
  }
  const risk = riskResult.data

  const decisionResult = await postJsonWithRetry<DecisionResponse>("/v1/decision", {
    customer_ref: order.customer_id ?? "unknown",
    order_ref: data.id,
    fulfillment_risk_score: risk.risk_score,
    model_versions: { fulfillment_risk: risk.model_artifact_sha256.slice(0, 12) },
  })

  const fulfillmentRiskRecord = {
    risk_score: risk.risk_score,
    risk_class: risk.risk_class,
    model_version: risk.model_version,
    model_experiment_id: risk.model_experiment_id,
    model_artifact_sha256: risk.model_artifact_sha256,
  }

  if (!decisionResult.ok) {
    const status = statusFromAiFailure(decisionResult)
    logger.warn(
      `[commercepilot-ai] order ${data.id}: decision engine call failed (${status}): ${decisionResult.error}`
    )
    await persist({
      idempotency_key: idempotencyKey,
      processing_status: status,
      order_id: data.id,
      fulfillment_risk: fulfillmentRiskRecord,
      feature_source: featureSource,
      error: decisionResult.error,
      processed_at: new Date().toISOString(),
    })
    return
  }
  const decision = decisionResult.data

  await persist({
    idempotency_key: idempotencyKey,
    processing_status: "COMPLETED",
    order_id: data.id,
    fulfillment_risk: fulfillmentRiskRecord,
    decision: {
      action: decision.action,
      priority: decision.priority,
      reason_codes: decision.reason_codes,
      ruleset_version: decision.ruleset_version,
    },
    feature_source: featureSource,
    processed_at: new Date().toISOString(),
  })

  logger.info(
    `[commercepilot-ai] order ${data.id}: COMPLETED fulfillment_risk=${risk.risk_score.toFixed(4)} ` +
      `(${risk.risk_class}) -> decision=${decision.action} priority=${decision.priority} ` +
      `reason_codes=${decision.reason_codes.join(",")}`
  )

  // SHADOW MODE ONLY: scores the new production-parity Seller-SLA model
  // (see PRODUCTION_PARITY_MODEL_COMPARISON.json -- WEAK signal, mean AUC
  // ~0.555) purely for first-party feedback-dataset collection. This call
  // is fully independent of, and never gates, the V1 decision above: any
  // failure here is logged and swallowed, never surfaced to the customer or
  // to order processing, and its result is never read by the Decision
  // Engine or any automated action. Persisted under a SEPARATE metadata key
  // (`commercepilot_ai_shadow_seller_sla`) so it can never be confused with
  // the live V1 `commercepilot_ai` record above.
  try {
    const shadowFeatures = await resolveProductionParityFeatures(container, data.id)
    if (shadowFeatures.status === "AVAILABLE_NOW") {
      const shadowResult = await postJsonWithRetry<{
        seller_sla_breach_probability: number
        risk_level: string
        model_version: string
        model_artifact_sha256: string
        persisted: boolean
      }>("/v1/fulfillment/seller-sla-shadow", {
        order_ref: data.id,
        purchase_timestamp: shadowFeatures.purchaseTimestamp,
        n_items: shadowFeatures.nItems,
        n_distinct_products: shadowFeatures.nDistinctProducts,
        n_categories: shadowFeatures.nCategories,
        total_price: shadowFeatures.totalPrice,
        total_freight: shadowFeatures.totalFreight,
        weight_g: shadowFeatures.weightG,
        volume_cm3: shadowFeatures.volumeCm3,
        payment_value: shadowFeatures.paymentValue,
        same_zone: shadowFeatures.sameZone,
      })
      const existingAfterMainCall = (
        (await orderModuleService.retrieveOrder(data.id)).metadata ?? {}
      ) as Record<string, unknown>
      if (shadowResult.ok) {
        await orderModuleService.updateOrders(data.id, {
          metadata: {
            ...existingAfterMainCall,
            commercepilot_ai_shadow_seller_sla: {
              mode: "SHADOW",
              status: "COMPLETED",
              order_id: data.id,
              seller_sla_breach_probability: shadowResult.data.seller_sla_breach_probability,
              risk_level: shadowResult.data.risk_level,
              model_version: shadowResult.data.model_version,
              model_artifact_sha256: shadowResult.data.model_artifact_sha256,
              persisted_by_ai_service: shadowResult.data.persisted,
              processed_at: new Date().toISOString(),
            },
          },
        })
      } else {
        logger.warn(
          `[commercepilot-ai-shadow] order ${data.id}: shadow seller-SLA call failed: ${shadowResult.error}`
        )
        await orderModuleService.updateOrders(data.id, {
          metadata: {
            ...existingAfterMainCall,
            commercepilot_ai_shadow_seller_sla: {
              mode: "SHADOW",
              status: statusFromAiFailure(shadowResult),
              order_id: data.id,
              error: shadowResult.error,
              processed_at: new Date().toISOString(),
            },
          },
        })
      }
    } else {
      logger.info(
        `[commercepilot-ai-shadow] order ${data.id}: insufficient features for shadow scoring (${shadowFeatures.reason})`
      )
    }
  } catch (err) {
    // Structured, non-silent failure logging (mission-required): never a
    // bare swallow. This never rethrows -- shadow scoring must not affect
    // order placement in any way.
    logger.error(
      `[commercepilot-ai-shadow] order ${data.id}: unexpected error during shadow scoring, ` +
        `order processing unaffected: ${err instanceof Error ? err.stack ?? err.message : String(err)}`
    )
  }
}

export const config: SubscriberConfig = {
  event: "order.placed",
}
