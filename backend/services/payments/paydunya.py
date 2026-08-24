import hashlib
import hmac
import logging
import os

import requests


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

PAYDUNYA_MASTER_KEY = os.getenv(
    "PAYDUNYA_MASTER_KEY",
    ""
)

PAYDUNYA_PRIVATE_KEY = os.getenv(
    "PAYDUNYA_PRIVATE_KEY",
    ""
)

PAYDUNYA_TOKEN = os.getenv(
    "PAYDUNYA_TOKEN",
    ""
)

PAYDUNYA_MODE = os.getenv(
    "PAYDUNYA_MODE",
    "sandbox"
).lower()

APP_ENV = os.getenv(
    "APP_ENV",
    "development"
).lower()

PAYDUNYA_TIMEOUT = 15


if (
    PAYDUNYA_MODE == "production"
    and APP_ENV == "production"
):

    PAYDUNYA_BASE_URL = (
        "https://app.paydunya.com/api/v1"
    )

else:

    PAYDUNYA_BASE_URL = (
        "https://app.paydunya.com/sandbox-api/v1"
    )


# ============================================================
# HEADERS
# ============================================================

def _headers():

    return {
        "Content-Type": "application/json",
        "PAYDUNYA-MASTER-KEY": PAYDUNYA_MASTER_KEY,
        "PAYDUNYA-PRIVATE-KEY": PAYDUNYA_PRIVATE_KEY,
        "PAYDUNYA-TOKEN": PAYDUNYA_TOKEN,
    }


# ============================================================
# CONFIG CHECK
# ============================================================

def is_configured():

    configured = all([
        PAYDUNYA_MASTER_KEY,
        PAYDUNYA_PRIVATE_KEY,
        PAYDUNYA_TOKEN,
    ])

    logger.warning(
        "PayDunya configuration check: %s",
        "configured" if configured else "NOT CONFIGURED"
    )

    return configured


# ============================================================
# CREATE CHECKOUT INVOICE
# ============================================================

def create_checkout_invoice(
    order,
    business,
    customer,
    items,
):

    if APP_ENV != "production":

        raise RuntimeError(
            "PayDunya payments are disabled "
            "outside production."
        )

    if PAYDUNYA_MODE != "production":

        raise RuntimeError(
            "PayDunya production payments are disabled "
            "unless PAYDUNYA_MODE=production."
        )

    if not is_configured():

        raise RuntimeError(
            "PayDunya credentials are not configured."
        )

    callback_url = os.getenv(
        "PAYDUNYA_CALLBACK_URL",
        ""
    )

    return_url = os.getenv(
        "PAYDUNYA_RETURN_URL",
        ""
    )

    cancel_url = os.getenv(
        "PAYDUNYA_CANCEL_URL",
        ""
    )

    invoice_items = {}

    for index, item in enumerate(items):

        invoice_items[
            f"item_{index}"
        ] = {
            "name": str(
                item.get("name", "")
            ),
            "quantity": int(
                item.get("quantity", 1)
            ),
            "unit_price": str(
                item.get("price", 0)
            ),
            "total_price": str(
                item.get("subtotal", 0)
            ),
            "description": str(
                item.get("description", "")
            ),
        }

    invoice = {
        "total_amount": float(
            order.total_price
        ),
        "description": (
            f"Payment for Order #{order.id}"
        ),
        "items": invoice_items,
        "customer": {
            "name": (
                customer.name
                or "Customer"
            ),
            "phone": (
                customer.phone
                or ""
            ),
        },
    }

    actions = {}

    if callback_url:
        actions["callback_url"] = callback_url

    if return_url:
        actions["return_url"] = return_url

    if cancel_url:
        actions["cancel_url"] = cancel_url

    payload = {
        "invoice": invoice,

        "store": {
            "name": (
                business.name
                or "Restaurant"
            ),
            "phone": (
                getattr(
                    business,
                    "phone",
                    ""
                )
                or ""
            ),
            "postal_address": (
                getattr(
                    business,
                    "address",
                    ""
                )
                or ""
            ),
        },

        "custom_data": {
            "order_id": str(
                order.id
            ),
            "business_id": str(
                business.id
            ),
            "customer_id": str(
                customer.id
            ),
        },
    }

    if actions:
        payload["actions"] = actions

    url = (
        PAYDUNYA_BASE_URL
        + "/checkout-invoice/create"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            headers=_headers(),
            timeout=PAYDUNYA_TIMEOUT,
        )

    except requests.RequestException as exc:

        logger.exception(
            "PAYDUNYA HTTP REQUEST EXCEPTION: %s",
            str(exc)
        )

        raise RuntimeError(
            f"PayDunya connection failed: {str(exc)}"
        )

    logger.warning(
        "========== PAYDUNYA RESPONSE %s ==========",
        response.status_code
    )

    logger.warning(
        "PayDunya response body: %s",
        response.text[:3000]
    )

    try:

        data = response.json()

    except ValueError:

        data = {}

        logger.error(
            "PayDunya returned non-JSON response."
        )

    if not response.ok:

        raise RuntimeError(
            f"PayDunya HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    logger.warning(
        "PayDunya parsed response: %s",
        data
    )

    if data.get("response_code") != "00":

        raise RuntimeError(
            data.get(
                "response_text",
                "PayDunya invoice creation failed."
            )
        )

    checkout_url = data.get(
        "response_text"
    )

    token = data.get(
        "token"
    )

    if not checkout_url or not token:

        logger.error(
            "PayDunya response missing checkout URL or token."
        )

        raise RuntimeError(
            "PayDunya did not return a checkout URL or token."
        )

    logger.warning(
        "PayDunya invoice successfully created for order %s.",
        order.id
    )

    logger.warning(
        "PayDunya checkout URL received: %s",
        bool(checkout_url)
    )

    logger.warning(
        "PayDunya payment token received: %s",
        bool(token)
    )

    return {
        "success": True,
        "checkout_url": checkout_url,
        "token": token,
    }


# ============================================================
# CHECK PAYMENT STATUS
# ============================================================

def check_payment_status(token):

    logger.warning(
        "========== PAYDUNYA PAYMENT STATUS CHECK =========="
    )

    if not is_configured():

        raise RuntimeError(
            "PayDunya credentials are not configured."
        )

    if not token:

        raise ValueError(
            "PayDunya token is required."
        )

    url = (
        PAYDUNYA_BASE_URL
        + "/checkout-invoice/confirm/"
        + str(token)
    )

    logger.warning(
        "Checking PayDunya payment status."
    )

    response = requests.get(
        url,
        headers=_headers(),
        timeout=PAYDUNYA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    return data


# ============================================================
# VERIFY CALLBACK HASH
# ============================================================

def verify_callback_hash(received_hash):

    if not received_hash:

        return False

    if not PAYDUNYA_MASTER_KEY:

        return False

    expected_hash = hashlib.sha512(
        PAYDUNYA_MASTER_KEY.encode(
            "utf-8"
        )
    ).hexdigest()

    return hmac.compare_digest(
        str(received_hash),
        expected_hash,
    )