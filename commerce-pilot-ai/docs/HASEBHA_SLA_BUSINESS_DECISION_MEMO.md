# One-Page Decision Memo: The Shipping-Time Promise

**To**: HASEBHA store owner/operations
**From**: Engineering
**Re**: One 15-minute decision that unblocks the entire fulfillment-risk prediction feature
**Date**: 2026-08-22

## What we need you to decide

For each shipping option you already offer customers (e.g. "Cairo Same-City," "National
Standard," or whatever your actual option names are), tell us: **how many business days do
we promise, from order placement, until the item ships to the carrier?**

That's it. One number per shipping option.

## Where to find your starting point

You likely already wrote something like this when you set up shipping options in the Medusa
admin panel — check each shipping option's own name/description field. If it already says
something like "ships within 2 days" or "same-day dispatch," that IS the number we need; we
just need it in a structured field the system can read, not only in the customer-facing text.
If no such promise currently exists anywhere for a given option, that's fine — just tell us
"no promise yet" for that one and we'll leave it out until you set one.

## Why this takes 15 minutes, not a project

We are not asking you to design a new operations process. We are asking you to write down a
number you likely already know informally (or to say "we haven't decided this yet" for any
option where you don't). The engineering side (a `promise_business_days` field per shipping
option) is already fully specified and ready to implement the moment you give us the numbers
— see `reports/generated/olist_v3_multistage/HASEBHA_SHIPPING_SLA_PRODUCT_REQUIREMENT.md` for
the technical detail, which you do not need to read.

## What this single decision unlocks

Right now, our fulfillment-risk prediction system has no defined meaning for "late" — there
is nothing to compare an order's shipping time against. Once this promise exists:

1. We can start correctly labeling real orders as "on-time" or "breached" the moment they
   ship — this is the exact data our prediction system needs to learn from your real store,
   instead of relying on a Brazilian public dataset that we have already proven does not
   transfer well to a single-vendor store like yours.
2. We start a **real, first-party** data-collection clock. We estimate needing roughly
   1,650 orders (minimum) to 4,500 orders (recommended) with a defined promise in place
   before we can build and validate a model on your own data — the sooner this decision is
   made, the sooner that clock starts.
3. Nothing changes for customers immediately — this decision only defines what "late" means
   internally for our own risk-monitoring; it does not change what promises you already show
   customers at checkout unless you separately decide to update that copy.

## What we are NOT asking you to decide right now

We are not asking you to approve any automated action (refunds, cancellations, changed
promises) based on this prediction system — that is permanently out of scope for this
project regardless of any future model's accuracy. We are only asking for the promise
number(s) so a real target can exist.

## Next step

Reply with the promise (in business days) for each active shipping option, or "not yet
decided" for any you want to skip for now. Engineering will implement the field and start the
data-collection clock the same day we receive it.
