import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

from langdetect import detect, DetectorFactory

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sqlalchemy.orm import load_only

from sqlalchemy.exc import IntegrityError

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    flash,
    url_for
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from database.db import db

from models.user import User
from models.business import Business
from models.menu import Menu
from models.order import Order
from models.payment import Payment
from models.order_item import OrderItem
from models.customer import Customer
from models.ingredient import Ingredient
from models.menu_ingredient import MenuIngredient
from models.customer_preference import CustomerPreference
from services.customer_memory import (learn_customer_preferences_from_order, learn_explicit_customer_preference)
from models.conversation import Conversation
from models.pending_order import PendingOrder
from models.support_ticket import SupportTicket
from models.customer_interaction import CustomerInteraction
from models.inventory_reservation import InventoryReservation
from models.whatsapp_message import WhatsAppMessage
from services.inventory import reserve_inventory_for_order, consume_inventory_for_order, release_inventory_for_order
from threading import Thread

from services.ai.prompt_builder import build_restaurant_prompt
from services.ai.openai_provider import OpenAIProvider
from services.ai.voice_transcriber import (
    transcribe_audio
)


from services.ai.order_modifier import (
    interpret_order_request
)

from services.ai.order_executor import (
    execute_order_action
)

from services.ai.order_extractor import (
    extract_order
)

from services.ai.agent import (
    run_agent
)

from services.ai.agent import (
    run_agent
)

# ============================================================
# WHATSAPP BACKGROUND PROCESSING
# ============================================================

WHATSAPP_EXECUTOR = ThreadPoolExecutor(
    max_workers=2
)

def send_n8n_webhook_async(
    n8n_webhook_url,
    message_id,
    business_id,
    from_phone,
    message_type,
    text_body,
):

    try:

        app.logger.warning(
            "[n8n] Background webhook → %s",
            n8n_webhook_url
        )

        response = requests.post(
            n8n_webhook_url,
            json={
                "event": "whatsapp_message",
                "message_id": message_id,
                "business_id": business_id,
                "from_phone": from_phone,
                "message_type": message_type,
                "text": text_body,
            },
            timeout=3
        )

        if response.ok:
            app.logger.info(
                "[n8n] Webhook delivered: %s",
                response.status_code,
            )
        else:
            app.logger.warning(
                "[n8n] Webhook returned %s: %s",
                response.status_code,
                response.text[:500],
            )

    except Exception as exc:

        app.logger.warning(
            "[n8n] Optional webhook unavailable: %s",
            exc,
        )

WHATSAPP_IN_FLIGHT = set()

WHATSAPP_IN_FLIGHT_LOCK = Lock()


# ============================================================
# CUSTOMER LANGUAGE DETECTION
# ============================================================

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "ar": "Arabic",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ru": "Russian",
    "tr": "Turkish",
    "sw": "Swahili",
}


def detect_customer_language(
    message,
    fallback="English"
):

    if not message:
        return fallback

    text = str(message).strip()

    if not text:
        return fallback

        # Common short messages and restaurant-related phrases.
    short_phrases = {
        # English
        "hi": "English",
        "hello": "English",
        "hey": "English",
        "good morning": "English",
        "good afternoon": "English",
        "good evening": "English",
        "yes": "English",
        "yeah": "English",
        "yep": "English",
        "please": "English",
        "thanks": "English",
        "thank you": "English",
        "menu": "English",
        "price": "English",
        "prices": "English",
        "how much": "English",
        "order": "English",
        "delivery": "English",
        "do you deliver": "English",
        "do you deliver?": "English",
        "can you deliver": "English",
        "can you deliver?": "English",
        "do you offer delivery": "English",
        "do you offer delivery?": "English",
        "is there delivery": "English",
        "is there delivery?": "English",
        "where do you deliver": "English",
        "where do you deliver?": "English",
        "cancel": "English",

        # French
        "bonjour": "French",
        "bonsoir": "French",
        "bjr": "French",
        "bsr": "French",
        "re": "French",
        "salut": "French",
        "lut": "French",
        "coucou": "French",
        "cc": "French",
        "yop": "French",
        "yo": "French",
        "wesh": "French",
        "wesh la famille lt": "French",
        "tfk": "French",
        "sdk": "French",
        "t'as la forme": "French",
        "ta la forme": "French",
        "tas la forme": "French",
        "ça roule": "French",
        "ca roule": "French",
        "cv": "French",
        "koi29": "French",
        "merci": "French",
        "s'il vous plaît": "French",
        "s'il te plaît": "French",
        "je veux": "French",
        "je voudrais": "French",
        "combien": "French",
        "combien ça coûte": "French",
        "prix": "French",
        "commande": "French",
        "commander": "French",
        "livraison": "French",
        "annuler": "French",
        "oui": "French",
        "non": "French",

        # Spanish
        "hola": "Spanish",
        "buenos dias": "Spanish",
        "buenas": "Spanish",
        "gracias": "Spanish",
        "por favor": "Spanish",
        "quiero": "Spanish",
        "cuanto": "Spanish",
        "precio": "Spanish",
        "pedido": "Spanish",
        "entrega": "Spanish",

        # Portuguese
        "olá": "Portuguese",
        "ola": "Portuguese",
        "bom dia": "Portuguese",
        "obrigado": "Portuguese",
        "obrigada": "Portuguese",
        "por favor": "Portuguese",
        "quero": "Portuguese",
        "quanto": "Portuguese",
        "preço": "Portuguese",
        "pedido": "Portuguese",

        # Italian
        "ciao": "Italian",
        "buongiorno": "Italian",
        "grazie": "Italian",
        "per favore": "Italian",
        "voglio": "Italian",
        "quanto": "Italian",

        # German
        "hallo": "German",
        "guten morgen": "German",
        "gunten morgen": "German",
        "danke": "German",
        "bitte": "German",
        "ich möchte": "German",
        "wie viel": "German",
    }

    normalized = re.sub(
        r"[-–—]",
        " ",
        text.lower()
    )

    normalized = re.sub(
        r"[^\w\s]",
        "",
        normalized
    )

    normalized = " ".join(
        normalized.split()
    )

    normalized = " ".join(
        normalized.split()
    )

    if normalized in short_phrases:
        return short_phrases[normalized]

    # ========================================================
    # COMMON MULTILINGUAL RESTAURANT PHRASES
    # ========================================================

    french_phrases = (
        # Menu / browsing
        "je peux voir",
        "je voudrais voir",
        "je veux voir",
        "puis je voir",
        "montre moi",
        "montrez moi",
        "donne moi",
        "donnez moi",
        "voir le menu",
        "voir votre menu",
        "voir ton menu",
        "quel est le menu",
        "qu est ce que vous avez",
        "qu est ce qu il y a",
        "que proposez vous",
        "que servez vous",

        # Ordering
        "je voudrais",
        "je veux",
        "j'aimerais",
        "j aimerais",
        "jaimerais",
        "je souhaite",
        "je vais prendre",
        "je prends",
        "je peux avoir",
        "je peux commander",
        "je voudrais commander",
        "je veux commander",
        "je souhaite commander",
        "donnez moi",
        "donne moi",

        # French quantity/order patterns
        "un ",
        "une ",
        "deux ",
        "trois ",
        "quatre ",
        "cinq ",
        "six ",
        "sept ",
        "huit ",
        "neuf ",
        "dix ",

        # Restaurant questions
        "combien ça coûte",
        "combien coute",
        "avez vous",
        "avez-vous",
    )

    english_phrases = (
    "can i see",
    "i want to see",
    "i would like to see",
    "i would like to order",
    "i want to order",
    "i would like a",
    "i want a",
    "i would like an",
    "i want an",
    "i'd like to order",
    "i'd like a",
    "i'd like an",
    "can i order",
    "can i have",
    "can i get",
    "i will have",
    "i'll have",
    "i am ordering",
    "i'm ordering",
    "show me",
    "give me",
    "what do you have",
    "what do you serve",
    "what is on the menu",
    "what's on the menu",
    "how much",
)

    spanish_phrases = (
        "puedo ver",
        "quiero ver",
        "muestreme",
        "muestrame",
        "dame",
        "que tienen",
        "que tienen en el menu",
    )

    portuguese_phrases = (
        "posso ver",
        "quero ver",
        "mostre",
        "me mostre",
        "me de",
        "o que voces tem",
    )

    italian_phrases = (
        "posso vedere",
        "voglio vedere",
        "mostrami",
        "fammi vedere",
        "cosa avete",
    )

    german_phrases = (
        "kann ich",
        "zeig mir",
        "zeige mir",
        "was habt ihr",
        "was gibt es",
    )

    if any(
        phrase in normalized
        for phrase in french_phrases
    ):
        return "French"

    if any(
        phrase in normalized
        for phrase in english_phrases
    ):
        return "English"

    if any(
        phrase in normalized
        for phrase in spanish_phrases
    ):
        return "Spanish"

    if any(
        phrase in normalized
        for phrase in portuguese_phrases
    ):
        return "Portuguese"

    if any(
        phrase in normalized
        for phrase in italian_phrases
    ):
        return "Italian"

    if any(
        phrase in normalized
        for phrase in german_phrases
    ):
        return "German"

    # ========================================================
    # FAST LOCAL FALLBACK
    # ========================================================

    # Keep the customer's current language for short or
    # ambiguous messages. This avoids an expensive detector
    # call on the normal WhatsApp response path.
    if len(normalized.split()) <= 8:
        return fallback

    language_keywords = {
        "French": (
            "je ", "tu ", "vous ", "nous ", "avec ",
            "pour ", "dans ", "sur ", "une ", "un ",
            "des ", "les ", "est ", "sont ", "pas ",
            "mais ", "merci ", "bonjour ", "commande ",
            "livraison ", "voudrais ", "aimerais ",
        ),
        "Spanish": (
            "yo ", "quiero ", "puedo ", "para ",
            "con ", "una ", "uno ", "los ", "las ",
            "gracias ", "hola ", "pedido ", "entrega ",
        ),
        "Portuguese": (
            "eu ", "quero ", "posso ", "para ",
            "com ", "uma ", "um ", "os ", "as ",
            "obrigado ", "pedido ", "entrega ",
        ),
        "Italian": (
            "io ", "voglio ", "posso ", "per ",
            "con ", "una ", "uno ", "gli ", "le ",
            "grazie ", "ordine ",
        ),
        "German": (
            "ich ", "möchte ", "kann ", "für ",
            "mit ", "eine ", "ein ", "die ",
            "der ", "das ", "danke ",
        ),
    }

    scores = {
        language: sum(
            normalized.count(keyword)
            for keyword in keywords
        )
        for language, keywords in language_keywords.items()
    }

    best_language = max(
        scores,
        key=scores.get
    )

    if scores[best_language] >= 2:
        return best_language

    return fallback


app = Flask(__name__)

# ============================================================
# APPLICATION CONFIG
# ============================================================

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "dev-only-change-me"
)


database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///business.db"
)

# Render/PostgreSQL may provide a plain postgres URL.
# Explicitly use Psycopg 3 when PostgreSQL is detected.
if database_url.startswith(
    "postgres://"
):
    database_url = database_url.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )

elif database_url.startswith(
    "postgresql://"
):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )

app.config[
    "SQLALCHEMY_DATABASE_URI"
] = database_url

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ============================================================
# POSTGRES CONNECTION POOL
# ============================================================
#
# Keep database connections warm between WhatsApp requests.
# This avoids repeatedly establishing a new PostgreSQL connection
# and detects stale connections before using them.

# PostgreSQL connection-pool tuning is only valid for PostgreSQL.
# SQLite uses StaticPool/SingletonThreadPool and rejects these options.
if database_url.startswith("postgresql"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 5,
        "max_overflow": 2,
        "pool_timeout": 5,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }
else:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {}

app.config["SESSION_COOKIE_HTTPONLY"] = True

app.config["SESSION_COOKIE_SAMESITE"] = os.environ.get(
    "SESSION_COOKIE_SAMESITE",
    "Lax"
)

app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get(
        "SESSION_COOKIE_SECURE",
        "false"
    ).lower() == "true"
)


db.init_app(app)


# ============================================================
# WHATSAPP CONFIG
# ============================================================

WHATSAPP_VERIFY_TOKEN = os.environ.get(
    "WHATSAPP_VERIFY_TOKEN",
    "my_verify_token"
)

WHATSAPP_TOKEN = os.environ.get(
    "WHATSAPP_TOKEN",
    ""
)

WHATSAPP_PHONE_NUMBER_ID = os.environ.get(
    "WHATSAPP_PHONE_NUMBER_ID",
    ""
)


# ============================================================
# SUPPORT / EMAIL CONFIG
# ============================================================

SUPPORT_EMAIL = os.environ.get(
    "SUPPORT_EMAIL",
    ""
)

SMTP_HOST = os.environ.get(
    "SMTP_HOST",
    ""
)

try:

    SMTP_PORT = int(
        os.environ.get(
            "SMTP_PORT",
            "587"
        )
    )

except (
    TypeError,
    ValueError
):

    SMTP_PORT = 587


SMTP_USERNAME = os.environ.get(
    "SMTP_USERNAME",
    ""
)

SMTP_PASSWORD = os.environ.get(
    "SMTP_PASSWORD",
    ""
)

SMTP_USE_TLS = (
    os.environ.get(
        "SMTP_USE_TLS",
        "true"
    ).lower() == "true"
)

ADMIN_EMAIL = os.environ.get(
    "ADMIN_EMAIL",
    SUPPORT_EMAIL
)


# ============================================================
# PAYMENT CONSTANTS
# ============================================================

PAYMENT_UNPAID = "Unpaid"
PAYMENT_PAID = "Paid"


# ============================================================
# WHATSAPP MESSAGE
# ============================================================

def send_whatsapp_message(
    to_phone,
    message_text
):

    """Send a text reply through WhatsApp Cloud API."""

    if (
        not WHATSAPP_TOKEN
        or not WHATSAPP_PHONE_NUMBER_ID
    ):

        app.logger.warning(
            "WhatsApp credentials are not configured."
        )

        return False

    try:

        import requests

        url = (
            f"https://graph.facebook.com/v23.0/"
            f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
        )

        _wa_start = time.perf_counter()

        response = requests.post(

            url,

            headers={
                "Authorization": (
                    f"Bearer {WHATSAPP_TOKEN}"
                ),
                "Content-Type": (
                    "application/json"
                )
            },

            json={
                "messaging_product": "whatsapp",
                "to": to_phone,
                "type": "text",
                "text": {
                    "body": message_text
                }
            },

            timeout=10
        )

        app.logger.warning(
            "[PERF WA] Graph API request: %.3fs status=%s",
            time.perf_counter() - _wa_start,
            response.status_code,
        )

        if response.ok:
            return True

        app.logger.error(
            "WhatsApp API error %s: %s",
            response.status_code,
            response.text[:1000]
        )

        return False

    except Exception:

        app.logger.exception(
            "WhatsApp message delivery failed."
        )

        return False

# ============================================================
# WHATSAPP AUDIO TRANSCRIPTION
# ============================================================

def download_whatsapp_audio(
    media_id
):
    """
    Download a WhatsApp audio file from Meta.
    Returns the local temporary file path.
    """

    if (
        not WHATSAPP_TOKEN
        or not media_id
    ):
        raise RuntimeError(
            "WhatsApp credentials or media ID are missing."
        )

    import os
    import tempfile
    import requests

    try:

        # ----------------------------------------------------
        # STEP 1: GET MEDIA URL
        # ----------------------------------------------------

        media_url = (
            f"https://graph.facebook.com/v23.0/"
            f"{media_id}"
        )

        media_response = requests.get(
            media_url,
            headers={
                "Authorization":
                    f"Bearer {WHATSAPP_TOKEN}"
            },
            timeout=10
        )

        if not media_response.ok:

            app.logger.error(
                "WhatsApp media lookup failed %s: %s",
                media_response.status_code,
                media_response.text[:1000]
            )

            raise RuntimeError(
                "Could not retrieve WhatsApp audio."
            )

        media_data = (
            media_response.json()
        )

        download_url = (
            media_data.get("url")
        )

        if not download_url:

            raise RuntimeError(
                "WhatsApp did not return an audio URL."
            )

        # ----------------------------------------------------
        # STEP 2: DOWNLOAD AUDIO
        # ----------------------------------------------------

        audio_response = requests.get(
            download_url,
            headers={
                "Authorization":
                    f"Bearer {WHATSAPP_TOKEN}"
            },
            timeout=30
        )

        if not audio_response.ok:

            app.logger.error(
                "WhatsApp audio download failed %s: %s",
                audio_response.status_code,
                audio_response.text[:1000]
            )

            raise RuntimeError(
                "Could not download WhatsApp audio."
            )

        # ----------------------------------------------------
        # STEP 3: SAVE TEMPORARILY
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".ogg"
        )

        temp_file.write(
            audio_response.content
        )

        temp_file.close()

        return temp_file.name

    except Exception:

        app.logger.exception(
            "WhatsApp audio download failed."
        )

        raise


def download_whatsapp_image(
    media_id
):
    """
    Download a WhatsApp image from Meta.

    Returns:
        tuple:
            image_bytes,
            mime_type
    """

    if (
        not WHATSAPP_TOKEN
        or not media_id
    ):
        raise RuntimeError(
            "WhatsApp credentials or media ID are missing."
        )

    try:

        media_url = (
            f"https://graph.facebook.com/v23.0/"
            f"{media_id}"
        )

        media_response = requests.get(
            media_url,
            headers={
                "Authorization":
                    f"Bearer {WHATSAPP_TOKEN}"
            },
            timeout=10
        )

        if not media_response.ok:

            app.logger.error(
                "WhatsApp image media lookup failed %s: %s",
                media_response.status_code,
                media_response.text[:1000]
            )

            raise RuntimeError(
                "Could not retrieve WhatsApp image."
            )

        media_data = (
            media_response.json()
        )

        download_url = (
            media_data.get("url")
        )

        mime_type = (
            media_data.get(
                "mime_type",
                "image/jpeg"
            )
        )

        if not download_url:

            raise RuntimeError(
                "WhatsApp did not return an image URL."
            )

        image_response = requests.get(
            download_url,
            headers={
                "Authorization":
                    f"Bearer {WHATSAPP_TOKEN}"
            },
            timeout=30
        )

        if not image_response.ok:

            app.logger.error(
                "WhatsApp image download failed %s: %s",
                image_response.status_code,
                image_response.text[:1000]
            )

            raise RuntimeError(
                "Could not download WhatsApp image."
            )

        if not image_response.content:

            raise RuntimeError(
                "WhatsApp returned an empty image."
            )

        return (
            image_response.content,
            mime_type
        )

    except Exception:

        app.logger.exception(
            "WhatsApp image download failed."
        )

        raise


def process_whatsapp_audio(
    media_id
):
    """
    Download and transcribe a WhatsApp voice message.

    Returns the transcribed text.
    """

    import os
    import shutil

    audio_path = None

    try:

        audio_path = download_whatsapp_audio(
            media_id
        )

        shutil.copy(
            audio_path,
            "/tmp/whatsapp-test.ogg"
        )

        app.logger.info(
            "Saved WhatsApp audio test file: %s",
            audio_path
        )

        transcript = transcribe_audio(
            audio_path
        )

        return transcript.strip()


    finally:

        if audio_path:

            try:
                os.remove(audio_path)

            except OSError:
                pass

def transcribe_audio(
    audio_path
):
    """
    Transcribe WhatsApp audio using Groq Whisper.
    """

    import os

    from groq import Groq

    if not audio_path:
        raise RuntimeError(
            "Audio file path is missing."
        )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    try:

        client = Groq(
            api_key=api_key
        )

        app.logger.info(
            "[VOICE] Starting Groq transcription: %s",
            audio_path
        )

        with open(
            audio_path,
            "rb"
        ) as audio_file:

            transcription = (
                client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="json",
                )
            )

        text = (
            getattr(
                transcription,
                "text",
                ""
            )
            or ""
        ).strip()

        if not text:

            raise RuntimeError(
                "Groq returned an empty transcription."
            )

        app.logger.info(
            "[VOICE] Groq transcript: %s",
            text
        )

        return text

    except Exception as exc:

        app.logger.exception(
            "[VOICE] Groq transcription failed: %s",
            exc
        )

        raise

# ============================================================
# SUPPORT EMAIL
# ============================================================

def send_support_email(
    recipient,
    subject,
    body,
    reply_to=None
):

    """
    Sends support email when SMTP credentials are configured.
    Email failure never removes an already-saved ticket.
    """

    if (
        not SMTP_HOST
        or not SMTP_USERNAME
        or not SMTP_PASSWORD
    ):

        app.logger.warning(
            "Support email skipped: SMTP is not configured."
        )

        return False

    try:

        import smtplib

        from email.message import EmailMessage

        msg = EmailMessage()

        msg["Subject"] = subject
        msg["From"] = SMTP_USERNAME
        msg["To"] = recipient

        if reply_to:

            msg["Reply-To"] = reply_to

        msg.set_content(body)

        with smtplib.SMTP(
            SMTP_HOST,
            SMTP_PORT,
            timeout=15
        ) as smtp:

            if SMTP_USE_TLS:

                smtp.starttls()

            smtp.login(
                SMTP_USERNAME,
                SMTP_PASSWORD
            )

            smtp.send_message(msg)

        return True

    except Exception:

        app.logger.exception(
            "Support email delivery failed."
        )

        return False


# ============================================================
# PAYMENT HELPERS
# ============================================================

def is_order_paid(order):

    """
    Returns True only when the order has actually been paid.
    """

    return (
        getattr(
            order,
            "payment_status",
            PAYMENT_UNPAID
        ) == PAYMENT_PAID
    )


def is_order_revenue_eligible(order):

    """
    An order contributes to revenue only when:

    1. Payment status is Paid
    2. Order has not been cancelled
    """

    return (
        is_order_paid(order)
        and order.status != "Cancelled"
    )


def paid_orders_for_business(orders):

    """
    Filter a collection of orders down to orders that
    should contribute to revenue.
    """

    return [
        order
        for order in orders
        if is_order_revenue_eligible(order)
    ]


def mark_order_as_paid(
    order,
    payment_method=None,
    transaction_id=None
):

    """
    Mark an order as paid.

    This is the central payment state transition.
    """

    order.payment_status = PAYMENT_PAID

    order.paid_at = datetime.now(timezone.utc)

    if payment_method:

        order.payment_method = (
            payment_method.strip()
        )

    if transaction_id:

        order.payment_transaction_id = (
            transaction_id.strip()
        )

    return order


def mark_order_as_unpaid(order):

    """
    Revert an order to unpaid.

    Normally this should only be used for payment corrections.
    """

    order.payment_status = PAYMENT_UNPAID
    order.paid_at = None
    order.payment_method = None
    order.payment_transaction_id = None

    return order


# ============================================================
# CENTRAL AI AGENT HELPER
# ============================================================

def run_customer_agent(
    business,
    phone,
    message,
    customer_name=None,
    image_context=None,
):

    """
    Find/create customer, load recent memory, run central
    AI agent and save conversation.
    """

    _rca_start = time.perf_counter()

    customer = Customer.query.filter_by(
        phone=phone,
        business_id=business.id
    ).first()

    app.logger.warning(
        "[PERF RCA] customer lookup: %.3fs",
        time.perf_counter() - _rca_start,
    )

    if not customer:

        customer = Customer(
            name=customer_name or "New Customer",
            phone=phone,
            language="English",
            business_id=business.id
        )

        db.session.add(customer)

        db.session.flush()
        
    elif (
        customer_name
        and (
            not customer.name
            or customer.name == "New Customer"
        )
    ):
        customer.name = customer_name


    # ========================================================
    # DETECT CURRENT CUSTOMER LANGUAGE
    # ========================================================

    _rca_stage = time.perf_counter()

    detected_language = detect_customer_language(
        message,
        fallback=customer.language or "English"
    )

    app.logger.warning(
        "[PERF RCA] language detection: %.3fs",
        time.perf_counter() - _rca_stage,
    )

    if detected_language != customer.language:

        customer.language = detected_language
        
    _rca_stage = time.perf_counter()

    previous_conversations = (
        Conversation.query
        .options(
            load_only(
                Conversation.message,
                Conversation.response,
            )
        )
        .filter_by(
            customer_id=customer.id
        )
        .order_by(
            Conversation.id.desc()
        )
        .limit(10)
        .all()
    )

    app.logger.warning(
        "[PERF RCA] conversation history: %.3fs",
        time.perf_counter() - _rca_stage,
    )

    history = [
        {
            "message": chat.message,
            "response": chat.response
        }
        for chat in reversed(
            previous_conversations
        )
    ]
    _rca_stage = time.perf_counter()

    result = run_agent(
        business.id,
        customer.phone,
        message,
        detected_language,
        history,
        image_context,
    )

    app.logger.warning(
        "[PERF RCA] run_agent: %.3fs",
        time.perf_counter() - _rca_stage,
    )

    if not isinstance(
        result,
        dict
    ):

        result = {
            "type": "response",
            "message": str(result)
        }

    response_message = (

        result.get("message")

        or result.get("response")

        or "I couldn't process that request right now."
    )

    conversation = Conversation(
        customer_id=customer.id,
        message=message,
        response=response_message
    )

    db.session.add(conversation)

    _rca_stage = time.perf_counter()

    db.session.commit()

    app.logger.warning(
        "[PERF RCA] conversation commit: %.3fs",
        time.perf_counter() - _rca_stage,
    )

    app.logger.warning(
        "[PERF RCA] TOTAL run_customer_agent: %.3fs",
        time.perf_counter() - _rca_start,
    )

    return (
        result,
        response_message,
        customer
    )


# ============================================================
# FIND WHATSAPP BUSINESS
# ============================================================

def find_whatsapp_business(value):

    """
    Resolve business from Meta phone_number_id.

    Older databases/models without the WhatsApp column
    safely fall back to the first registered business.
    """

    metadata = (
        value.get(
            "metadata",
            {}
        )
        if isinstance(value, dict)
        else {}
    )

    phone_number_id = metadata.get(
        "phone_number_id"
    )

    column = getattr(
        Business,
        "whatsapp_phone_number_id",
        None
    )

    if (
        column is not None
        and phone_number_id
    ):

        business = Business.query.filter(
            column == str(
                phone_number_id
            )
        ).first()

        if business:

            return business

    return (
        Business.query
        .order_by(
            Business.id.asc()
        )
        .first()
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ============================================================
# RESTAURANT AI DEMO
# ============================================================

from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired,
)


DEMO_RESTAURANT_NAME = (
    "Italian Ice Cream Cornetto"
)

DEMO_TRIAL_SECONDS = (
    4 * 24 * 60 * 60
)


def get_demo_business():
    return (
        Business.query
        .filter_by(
            name=DEMO_RESTAURANT_NAME
        )
        .first()
    )


def get_demo_serializer():

    return URLSafeTimedSerializer(
        app.secret_key,
        salt="botify-demo-trial",
    )


def create_demo_trial_token(
    business_id
):

    serializer = get_demo_serializer()

    return serializer.dumps(
        {
            "business_id": business_id
        }
    )


def validate_demo_trial_token(
    token
):

    serializer = get_demo_serializer()

    try:

        data = serializer.loads(
            token,
            max_age=DEMO_TRIAL_SECONDS,
        )

    except SignatureExpired:

        return None

    except BadSignature:

        return None

    if not isinstance(
        data,
        dict
    ):
        return None

    return data


def demo_trial_expired():

    expires_at = session.get(
        "demo_trial_expires_at"
    )

    if not expires_at:

        return False

    import time

    return (
        time.time()
        >= float(expires_at)
    )


@app.route("/demo")
def demo():

    business = get_demo_business()

    if not business:

        return (
            "Demo restaurant not found.",
            404
        )

    token = create_demo_trial_token(
        business.id
    )

    return redirect(
        url_for(
            "demo_trial",
            token=token
        )
    )


@app.route(
    "/demo/trial/<token>"
)
def demo_trial(
    token
):

    trial_data = (
        validate_demo_trial_token(
            token
        )
    )

    if not trial_data:

        return render_template(
            "demo_expired.html"
        ), 410

    business = db.session.get(
        Business,
        int(
            trial_data[
                "business_id"
            ]
        )
    )

    if not business:

        return (
            "Demo restaurant not found.",
            404
        )

    import time

    issued_at = time.time()

    session["demo_business_id"] = (
        business.id
    )

    session["demo_trial_expires_at"] = (
        issued_at
        + DEMO_TRIAL_SECONDS
    )

    return render_template(
        "demo.html",
        business=business
    )


@app.route(
    "/demo/chat",
    methods=["POST"]
)
def demo_chat():

    if demo_trial_expired():

        session.pop(
            "demo_phone",
            None
        )

        return {
            "success": False,
            "expired": True,
            "message": (
                "This demo trial has expired."
            ),
        }, 410

    data = request.get_json(
        silent=True
    ) or {}

    message = str(
        data.get(
            "message",
            ""
        )
    ).strip()

    if not message:

        return {
            "success": False,
            "message": (
                "Please enter a message."
            ),
        }, 400

    business_id = session.get(
        "demo_business_id"
    )

    if not business_id:

        return {
            "success": False,
            "message": (
                "Demo session expired. "
                "Please reopen the demo link."
            ),
        }, 410

    business = db.session.get(
        Business,
        int(
            business_id
        )
    )

    if not business:

        return {
            "success": False,
            "message": (
                "Demo restaurant not found."
            ),
        }, 404

    demo_phone = session.get(
        "demo_phone"
    )

    if not demo_phone:

        import uuid

        demo_phone = (
            "DEMO-"
            + uuid.uuid4().hex[:12]
        )

        session["demo_phone"] = (
            demo_phone
        )

    try:

        (
            agent_result,
            reply_text,
            customer,
        ) = run_customer_agent(
            business,
            demo_phone,
            message,
            "Demo Customer",
        )

        return {
            "success": True,
            "message": reply_text,
        }

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Demo AI request failed."
        )

        return {
            "success": False,
            "message": (
                "The demo assistant could not "
                "process that message."
            ),
        }, 500


@app.route(
    "/demo/reset",
    methods=["POST"]
)
def demo_reset():

    demo_phone = session.get(
        "demo_phone"
    )

    business_id = session.get(
        "demo_business_id"
    )

    if (
        demo_phone
        and business_id
    ):

        try:

            customer = (
                Customer.query
                .filter_by(
                    phone=demo_phone,
                    business_id=business_id,
                )
                .first()
            )

            if customer:

                PendingOrder.query.filter_by(
                    customer_id=customer.id,
                    business_id=business_id,
                ).delete(
                    synchronize_session=False
                )

                Conversation.query.filter_by(
                    customer_id=customer.id
                ).delete(
                    synchronize_session=False
                )

                demo_orders = (
                    Order.query
                    .filter_by(
                        customer_id=customer.id,
                        business_id=business_id,
                    )
                    .all()
                )

                for order in demo_orders:

                    order.status = "Cancelled"

                    order.payment_status = (
                        "Unpaid"
                    )

                    order.payment_token = None

                    order.payment_transaction_id = (
                        None
                    )

                    order.paid_at = None

                db.session.commit()

        except Exception:

            db.session.rollback()

            app.logger.exception(
                "Demo state reset failed."
            )

            return {
                "success": False,
                "message": (
                    "Could not reset the demo."
                ),
            }, 500

    session.pop(
        "demo_phone",
        None
    )

    return {
        "success": True,
        "message": (
            "Demo reset successfully."
        ),
    }

# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not email or not password:

            flash(
                "Enter your email address and password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        user = User.query.filter_by(
            email=email
        ).first()

        if (
            user
            and check_password_hash(
                user.password,
                password
            )
        ):

            session.clear()

            session["user_id"] = user.id

            flash(
                f"Welcome back, {user.username}.",
                "success"
            )

            return redirect(
                "/dashboard"
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        language = request.form.get(
            "language",
            "English"
        ).strip() or "English"

        if len(username) < 2:

            flash(
                "Username must be at least 2 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if not email or "@" not in email:

            flash(
                "Enter a valid email address.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 6:

            flash(
                "Password must be at least 6 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if User.query.filter_by(
            username=username
        ).first():

            flash(
                "That username is already in use.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if User.query.filter_by(
            email=email
        ).first():

            flash(
                "An account with that email already exists.",
                "error"
            )

            return render_template(
                "register.html"
            )

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(
                password
            ),
            language=language
        )

        try:

            db.session.add(user)

            db.session.commit()

        except Exception:

            db.session.rollback()

            flash(
                "We couldn't create your account. Please try again.",
                "error"
            )

            return render_template(
                "register.html"
            )

        flash(
            "Account created successfully. Sign in to continue.",
            "success"
        )

        return redirect(
            "/login"
        )

    return render_template(
        "register.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    businesses = Business.query.filter_by(
        owner_id=session["user_id"]
    ).all()

    return render_template(
        "dashboard.html",
        businesses=businesses
    )


# ============================================================
# CREATE BUSINESS
# ============================================================

@app.route(
    "/create-business",
    methods=["GET", "POST"]
)
def create_business():

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    if request.method == "POST":

        business = Business(

            name=request.form.get(
                "name"
            ),

            business_type=request.form.get(
                "business_type"
            ),

            address=request.form.get(
                "address"
            ),

            phone=request.form.get(
                "phone"
            ),

            owner_id=session["user_id"]
        )

        db.session.add(
            business
        )

        db.session.commit()

        return redirect(
            "/dashboard"
        )

    return render_template(
        "create_business.html"
    )


# ============================================================
# ORDER MANAGEMENT HELPERS
# ============================================================

def recalculate_order_total(order):

    total = 0

    for item in order.items:

        quantity = int(
            item.quantity or 0
        )

        price = float(
            item.price or 0
        )

        item.subtotal = (
            price * quantity
        )

        total += item.subtotal

    order.total_price = total

    return total


def order_can_be_modified(order):

    return order.status in {
        "Pending",
        "Preparing"
    }


def order_can_be_cancelled(order):

    return order.status not in {
        "Completed",
        "Cancelled"
    }


# ============================================================
# BUSINESS DASHBOARD
# ============================================================

@app.route(
    "/business/<int:business_id>"
)
def business_dashboard(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    # --------------------------------------------------------
    # MENU
    # --------------------------------------------------------

    menu_items = Menu.query.filter_by(
        business_id=business.id
    ).all()

    # --------------------------------------------------------
    # ORDERS
    # --------------------------------------------------------

    orders = (
        Order.query
        .filter_by(
            business_id=business.id
        )
        .order_by(
            Order.id.desc()
        )
        .all()
    )

    recent_orders = orders[:8]

    # --------------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------------

    customers = Customer.query.filter_by(
        business_id=business.id
    ).all()

    # --------------------------------------------------------
    # PAYMENT-AWARE ANALYTICS
    # --------------------------------------------------------

    paid_orders = paid_orders_for_business(
        orders
    )

    total_orders = len(
        orders
    )

    total_customers = len(
        customers
    )

    total_revenue = sum(

        float(
            order.total_price or 0
        )

        for order in paid_orders
    )

    pending_orders = sum(
        1
        for order in orders
        if order.status == "Pending"
    )

    preparing_orders = sum(
        1
        for order in orders
        if order.status == "Preparing"
    )

    ready_orders = sum(
        1
        for order in orders
        if order.status == "Ready"
    )

    completed_orders = sum(
        1
        for order in orders
        if order.status == "Completed"
    )

    cancelled_orders = sum(
        1
        for order in orders
        if order.status == "Cancelled"
    )

    unpaid_orders = sum(
        1
        for order in orders
        if not is_order_paid(order)
        and order.status != "Cancelled"
    )

    paid_order_count = len(
        paid_orders
    )

    active_orders = (
        pending_orders
        + preparing_orders
        + ready_orders
    )

    average_order_value = (

        total_revenue / paid_order_count

        if paid_order_count

        else 0
    )

    # --------------------------------------------------------
    # TOP SELLING ITEMS
    # --------------------------------------------------------

    item_sales = {}

    for order in paid_orders:

        for item in order.items:

            name = item.name

            if name not in item_sales:

                item_sales[name] = {

                    "name": name,

                    "quantity": 0,

                    "revenue": 0
                }

            item_sales[name][
                "quantity"
            ] += int(
                item.quantity or 0
            )

            item_sales[name][
                "revenue"
            ] += float(
                item.subtotal or 0
            )

    top_items = sorted(

        item_sales.values(),

        key=lambda item:
            item["quantity"],

        reverse=True

    )[:5]

    # --------------------------------------------------------
    # ORDER STATUS
    # --------------------------------------------------------

    order_status = {

        "Pending":
            pending_orders,

        "Preparing":
            preparing_orders,

        "Ready":
            ready_orders,

        "Completed":
            completed_orders,

        "Cancelled":
            cancelled_orders
    }

    # --------------------------------------------------------
    # 7-DAY REVENUE
    # --------------------------------------------------------

    today = datetime.now(timezone.utc).date()

    revenue_chart = []

    for days_ago in range(
        6,
        -1,
        -1
    ):

        chart_date = (
            today
            - timedelta(
                days=days_ago
            )
        )

        daily_revenue = 0

        for order in paid_orders:

            if not order.created_at:

                continue

            if (
                order.created_at.date()
                == chart_date
            ):

                daily_revenue += float(
                    order.total_price or 0
                )

        revenue_chart.append({

            "date":
                chart_date.strftime("%a"),

            "full_date":
                chart_date.strftime("%d %b"),

            "revenue":
                daily_revenue
        })

    return render_template(

        "business_dashboard.html",

        business=business,

        menu_items=menu_items,

        orders=orders,

        recent_orders=recent_orders,

        customers=customers,

        total_orders=total_orders,

        total_customers=total_customers,

        total_revenue=total_revenue,

        pending_orders=pending_orders,

        preparing_orders=preparing_orders,

        ready_orders=ready_orders,

        completed_orders=completed_orders,

        cancelled_orders=cancelled_orders,

        unpaid_orders=unpaid_orders,

        paid_order_count=paid_order_count,

        active_orders=active_orders,

        average_order_value=average_order_value,

        top_items=top_items,

        revenue_chart=revenue_chart,

        order_status=order_status,

        revenue=total_revenue
    )


# ============================================================
# MENU INTELLIGENCE API
# ============================================================

@app.route(
    "/business/<int:business_id>/menu/search"
)
def menu_search_api(
    business_id
):

    if "user_id" not in session:

        return {
            "error": "Unauthorized"
        }, 401

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return {
            "error": "Business not found"
        }, 404

    from services.ai.menu_intelligence import (
        search_menu,
        recommend_menu
    )

    query = request.args.get(
        "q",
        ""
    ).strip()

    if query:

        results = search_menu(
            business.id,
            query,
            limit=5
        )

    else:

        results = recommend_menu(
            business.id,
            limit=5
        )

    return {
        "business_id": business.id,
        "query": query,
        "results": results,
        "count": len(results)
    }


# ============================================================
# MODIFY ORDER
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/modify",
    methods=["POST"]
)
def modify_order(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    order = Order.query.filter_by(
        id=order_id,
        business_id=business.id
    ).first()

    if not order:

        return "Order not found", 404

    if not order_can_be_modified(
        order
    ):

        return (
            "This order can no longer be modified.",
            400
        )

    item_id = request.form.get(
        "item_id"
    )

    action = request.form.get(
        "action",
        "update"
    )

    try:

        item_id = int(
            item_id
        )

    except (
        TypeError,
        ValueError
    ):

        return "Invalid order item.", 400

    item = OrderItem.query.filter_by(
        id=item_id,
        order_id=order.id
    ).first()

    if not item:

        return "Order item not found.", 404

    if action == "remove":

        db.session.delete(
            item
        )

        db.session.flush()

        remaining_items = (
            OrderItem.query
            .filter_by(
                order_id=order.id
            )
            .all()
        )

        if remaining_items:

            recalculate_order_total(
                order
            )

        else:

            order.total_price = 0

    elif action == "update":

        quantity_raw = request.form.get(
            "quantity",
            ""
        ).strip()

        try:

            quantity = int(
                quantity_raw
            )

        except ValueError:

            return (
                "Quantity must be a whole number.",
                400
            )

        if quantity < 1:

            return (
                "Quantity must be at least 1.",
                400
            )

        item.quantity = quantity

        recalculate_order_total(
            order
        )

    else:

        return (
            "Invalid modification action.",
            400
        )

    db.session.commit()

    return redirect(
        f"/business/{business.id}/orders"
    )


# ============================================================
# CANCEL ORDER
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/cancel",
    methods=["POST"]
)
def cancel_order(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    order = Order.query.filter_by(
        id=order_id,
        business_id=business.id
    ).first()

    if not order:

        return "Order not found", 404

    if not order_can_be_cancelled(
        order
    ):

        return (
            "This order cannot be cancelled.",
            400
        )

    order.status = "Cancelled"

    db.session.commit()

    return redirect(
        f"/business/{business.id}/orders"
    )


# ============================================================
# VIEW ORDERS
# ============================================================

@app.route(
    "/business/<int:business_id>/orders"
)
def view_orders(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    orders = (
        Order.query
        .filter_by(
            business_id=business.id
        )
        .order_by(
            Order.id.desc()
        )
        .all()
    )

    return render_template(

        "orders.html",

        business=business,

        orders=orders
    )


# ============================================================
# UPDATE ORDER STATUS
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/status",
    methods=["POST"]
)
def update_order_status(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found.", 404

    order = Order.query.filter_by(
        id=order_id,
        business_id=business.id
    ).first()

    if not order:

        return "Order not found.", 404

    new_status = request.form.get(
        "status",
        ""
    ).strip()

    allowed_statuses = {

        "Pending",

        "Preparing",

        "Ready",

        "Completed",

        "Cancelled"
    }

    if new_status not in allowed_statuses:

        return "Invalid order status.", 400

    old_status = order.status

    # Prevent inventory from being consumed/released twice
    # or a terminal order from being moved backwards.
    if old_status == "Completed" and new_status != "Completed":

        return (
            "Completed orders cannot be moved to another status.",
            400,
        )

    if old_status == "Cancelled" and new_status != "Cancelled":

        return (
            "Cancelled orders cannot be moved to another status.",
            400,
        )

    # ==========================================
    # INVENTORY STATE TRANSITION
    # ==========================================

    if new_status == "Completed" and old_status != "Completed":

        try:
            consume_inventory_for_order(order.id)

        except Exception:

            db.session.rollback()

            logger.exception(
                "Inventory consumption failed for order %s.",
                order.id,
            )

            return (
                "Could not complete the order because "
                "inventory could not be finalized.",
                500,
            )

    elif new_status == "Cancelled" and old_status != "Cancelled":

        try:
            release_inventory_for_order(order.id)

        except Exception:

            db.session.rollback()

            logger.exception(
                "Inventory release failed for order %s.",
                order.id,
            )

            return (
                "Could not cancel the order because "
                "inventory could not be released.",
                500,
            )

    order.status = new_status

    # Learn customer preferences only after the order is completed.
    if new_status == "Completed":

        try:
            learn_customer_preferences_from_order(order)

        except Exception:
            logger.exception(
                "Customer preference learning failed for order %s.",
                order.id,
            )

    db.session.commit()

    return redirect(
        f"/business/{business.id}/orders"
    )


# ============================================================
# MARK ORDER AS PAID
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/pay",
    methods=["POST"]
)
def mark_order_paid(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found.", 404

    order = Order.query.filter_by(
        id=order_id,
        business_id=business.id
    ).first()

    if not order:

        return "Order not found.", 404

    if order.status == "Cancelled":

        return (
            "A cancelled order cannot be marked as paid.",
            400
        )

    payment_method = request.form.get(
        "payment_method",
        "Manual"
    ).strip()

    transaction_id = request.form.get(
        "payment_transaction_id",
        ""
    ).strip()

    mark_order_as_paid(

        order,

        payment_method=payment_method,

        transaction_id=transaction_id
    )

    db.session.commit()

    flash(
        f"Order #{order.id} marked as paid.",
        "success"
    )

    return redirect(
        f"/business/{business.id}/orders"
    )


# ============================================================
# MARK ORDER AS UNPAID
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/unpay",
    methods=["POST"]
)
def mark_order_unpaid(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found.", 404

    order = Order.query.filter_by(
        id=order_id,
        business_id=business.id
    ).first()

    if not order:

        return "Order not found.", 404

    mark_order_as_unpaid(
        order
    )

    db.session.commit()

    flash(
        f"Order #{order.id} marked as unpaid.",
        "success"
    )

    return redirect(
        f"/business/{business.id}/orders"
    )


# ============================================================
# CREATE MANUAL ORDER
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/create",
    methods=["GET", "POST"]
)
def create_order(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found"

    if request.method == "POST":

        order = Order(

            customer_name=request.form.get(
                "customer_name"
            ),

            customer_phone=request.form.get(
                "customer_phone"
            ),

            delivery_address=request.form.get(
                "delivery_address"
            ),

            total_price=float(
                request.form.get(
                    "total_price"
                )
            ),

            status="Pending",

            payment_status=PAYMENT_UNPAID,

            business_id=business.id
        )

        db.session.add(
            order
        )

        # ========================================================
        # N8N ORDER AUTOMATION
        # ========================================================

        n8n_webhook_url = os.environ.get(
            "N8N_WEBHOOK_URL"
        )

        if n8n_webhook_url:

            try:

                requests.post(
                    n8n_webhook_url,
                    json={
                        "event": "order_created",
                        "order_id": order.id,
                        "business_id": business.id,
                        "customer_name": order.customer_name,
                        "customer_phone": order.customer_phone,
                        "delivery_address": order.delivery_address,
                        "total_price": order.total_price,
                        "status": order.status,
                        "payment_status": order.payment_status,
                    },
                    timeout=5
                )

            except Exception as e:

                print(
                    f"[n8n] Order webhook failed: {e}"
                )

        db.session.commit()

        return redirect(
            f"/business/{business.id}/orders"
        )

    return render_template(
        "create_order.html",
        business=business
    )


# ============================================================
# CREATE MENU ITEM
# ============================================================

@app.route(
    "/business/<int:business_id>/menu/create",
    methods=["GET", "POST"]
)
def create_menu(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found"

    if request.method == "POST":

        menu = Menu(

            name=request.form.get(
                "name"
            ),

            description=request.form.get(
                "description"
            ),

            price=float(
                request.form.get(
                    "price"
                )
            ),

            category=request.form.get(
                "category"
            ),

            available=True,

            business_id=business.id
        )

        db.session.add(
            menu
        )

        db.session.commit()

        return redirect(
            f"/business/{business.id}"
        )

    return render_template(
        "create_menu.html",
        business=business
    )


# ============================================================
# EDIT MENU
# ============================================================

@app.route(
    "/menu/<int:menu_id>/edit",
    methods=["GET", "POST"]
)
def edit_menu(
    menu_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    menu = Menu.query.get_or_404(
        menu_id
    )

    business = Business.query.filter_by(
        id=menu.business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Unauthorized"

    if request.method == "POST":

        menu.name = request.form.get(
            "name"
        )

        menu.description = request.form.get(
            "description"
        )

        menu.price = float(
            request.form.get(
                "price"
            )
        )

        menu.category = request.form.get(
            "category"
        )

        db.session.commit()

        return redirect(
            f"/business/{business.id}"
        )

    return render_template(
        "edit_menu.html",
        menu=menu
    )


# ============================================================
# DELETE MENU
# ============================================================

@app.route(
    "/menu/<int:menu_id>/delete"
)
def delete_menu(
    menu_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    menu = Menu.query.get_or_404(
        menu_id
    )

    business = Business.query.filter_by(
        id=menu.business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Unauthorized"

    db.session.delete(
        menu
    )

    db.session.commit()

    return redirect(
        f"/business/{business.id}"
    )


# ============================================================
# CUSTOMERS CRM
# ============================================================

@app.route(
    "/business/<int:business_id>/customers",
    methods=["GET", "POST"]
)
def customers(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        language = request.form.get(
            "language",
            "English"
        ).strip()

        if not phone:

            return (
                "Phone number is required.",
                400
            )

        existing_customer = (
            Customer.query
            .filter_by(
                phone=phone,
                business_id=business.id
            )
            .first()
        )

        if existing_customer:

            return redirect(
                f"/business/{business.id}/customer/"
                f"{existing_customer.id}"
            )

        customer = Customer(

            name=name or None,

            phone=phone,

            language=language,

            business_id=business.id
        )

        db.session.add(
            customer
        )

        db.session.commit()

        return redirect(
            f"/business/{business.id}/customers"
        )

    search = request.args.get(
        "search",
        ""
    ).strip()

    query = Customer.query.filter_by(
        business_id=business.id
    )

    if search:

        query = query.filter(

            db.or_(

                Customer.name.ilike(
                    f"%{search}%"
                ),

                Customer.phone.ilike(
                    f"%{search}%"
                )
            )
        )

    customers_list = (
        query
        .order_by(
            Customer.id.desc()
        )
        .all()
    )

    from sqlalchemy import func

    order_stats = db.session.query(

        Order.customer_id,

        func.count(
            Order.id
        ).label(
            "order_count"
        ),

        func.coalesce(
            func.sum(
                db.case(
                    (
                        db.and_(
                            Order.payment_status == PAYMENT_PAID,
                            Order.status != "Cancelled"
                        ),
                        Order.total_price
                    ),
                    else_=0
                )
            ),
            0
        ).label(
            "total_spent"
        ),

        func.max(
            Order.id
        ).label(
            "last_order_id"
        )

    ).filter(

        Order.business_id == business.id

    ).group_by(

        Order.customer_id

    ).all()

    customer_stats = {

        row.customer_id: {

            "orders":
                row.order_count,

            "spent":
                float(
                    row.total_spent or 0
                ),

            "last_order_id":
                row.last_order_id
        }

        for row in order_stats
    }

    total_customers = (
        Customer.query
        .filter_by(
            business_id=business.id
        )
        .count()
    )

    customers_with_orders = len(
        customer_stats
    )

    total_revenue = sum(

        stats["spent"]

        for stats in customer_stats.values()
    )

    return render_template(

        "customers.html",

        business=business,

        customers=customers_list,

        search=search,

        total_customers=total_customers,

        customers_with_orders=customers_with_orders,

        total_revenue=total_revenue,

        customer_stats=customer_stats
    )


# ============================================================
# RESTAURANT SETTINGS
# ============================================================

@app.route(
    "/business/<int:business_id>/settings",
    methods=["GET", "POST"]
)
def business_settings(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        business_type = request.form.get(
            "business_type",
            ""
        ).strip()

        address = request.form.get(
            "address",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        opening_hours = request.form.get(
            "opening_hours",
            ""
        ).strip()

        delivery_policy = request.form.get(
            "delivery_policy",
            ""
        ).strip()

        if not name:

            return (
                "Business name is required.",
                400
            )

        if not business_type:

            return (
                "Business type is required.",
                400
            )

        business.name = name

        business.business_type = (
            business_type
        )

        business.address = (
            address or None
        )

        business.phone = (
            phone or None
        )

        business.phone = (
            phone or None
        )

        business.opening_hours = (
            opening_hours or None
        )

        business.delivery_policy = (
            delivery_policy or None
        )

        business.delivery_policy = (
            delivery_policy or None
        )

        db.session.commit()

        return redirect(
            f"/business/{business.id}/settings"
        )

    return render_template(

        "business_settings.html",

        business=business
    )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

@app.route(
    "/business/<int:business_id>/customer/<int:customer_id>"
)
def customer_profile(
    business_id,
    customer_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    customer = Customer.query.filter_by(
        id=customer_id,
        business_id=business.id
    ).first()

    if not customer:

        return "Customer not found", 404

    orders = (

        Order.query
        .filter_by(
            customer_id=customer.id,
            business_id=business.id
        )
        .order_by(
            Order.id.desc()
        )
        .limit(20)
        .all()
    )

    total_orders = (
        Order.query
        .filter_by(
            customer_id=customer.id,
            business_id=business.id
        )
        .count()
    )

    from sqlalchemy import func

    total_spent = db.session.query(

        func.coalesce(

            func.sum(

                db.case(
                    (
                        db.and_(
                            Order.payment_status == PAYMENT_PAID,
                            Order.status != "Cancelled"
                        ),
                        Order.total_price
                    ),
                    else_=0
                )
            ),

            0
        )

    ).filter(

        Order.customer_id == customer.id,

        Order.business_id == business.id

    ).scalar()

    conversations = (

        Conversation.query
        .filter_by(
            customer_id=customer.id
        )
        .order_by(
            Conversation.id.desc()
        )
        .limit(20)
        .all()
    )

    total_conversations = (
        Conversation.query
        .filter_by(
            customer_id=customer.id
        )
        .count()
    )

    latest_order = (
        orders[0]
        if orders
        else None
    )

    latest_conversation = (

        conversations[0]

        if conversations

        else None
    )

    return render_template(

        "customer_profile.html",

        business=business,

        customer=customer,

        orders=orders,

        conversations=conversations,

        total_orders=total_orders,

        total_spent=float(
            total_spent or 0
        ),

        total_conversations=total_conversations,

        latest_order=latest_order,

        latest_conversation=latest_conversation
    )


# ============================================================
# AI PLAYGROUND
# ============================================================

@app.route(
    "/business/<int:business_id>/ai",
    methods=["GET", "POST"]
)
def ai_playground(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    response = None

    customer = None

    conversations = []

    customer_phone = ""

    message = ""

    frontend_action = None

    if request.method == "POST":

        message = request.form.get(
            "message",
            ""
        ).strip()

        customer_phone = request.form.get(
            "phone",
            ""
        ).strip()

        if not message or not customer_phone:

            response = (
                "Please enter a customer phone number "
                "and message."
            )

        else:

            customer = Customer.query.filter_by(

                phone=customer_phone,

                business_id=business.id

            ).first()

            if not customer:

                customer = Customer(

                    name="New Customer",

                    phone=customer_phone,

                    language="English",

                    business_id=business.id
                )

                db.session.add(
                    customer
                )

                db.session.commit()

            previous_conversations = (

                Conversation.query
                .filter_by(
                    customer_id=customer.id
                )
                .order_by(
                    Conversation.id.desc()
                )
                .limit(12)
                .all()
            )

            history = [

                {
                    "message":
                        chat.message,

                    "response":
                        chat.response
                }

                for chat in reversed(
                    previous_conversations
                )
            ]

            try:

                agent_result = run_agent(

                    business.id,

                    customer.phone,

                    message,

                    customer.language,

                    history
                )

                if not isinstance(
                    agent_result,
                    dict
                ):

                    response = (
                        "I received an invalid "
                        "response from the AI agent."
                    )

                elif agent_result.get(
                    "type"
                ) == "frontend_action":

                    frontend_action = (
                        agent_result.get(
                            "action"
                        )
                    )

                    action_messages = {

                        "open_dashboard":
                            "Opening the business dashboard.",

                        "open_orders":
                            "Opening your orders.",

                        "open_customers":
                            "Opening your customers.",

                        "open_menu":
                            "Opening your menu.",

                        "open_ai":
                            "Opening the AI assistant.",

                        "open_settings":
                            "Opening restaurant settings."
                    }

                    response = action_messages.get(

                        frontend_action,

                        "Opening that section."
                    )

                else:

                    response = (

                        agent_result.get(
                            "message"
                        )

                        or
                        "I couldn't generate a response right now."
                    )

            except Exception:

                db.session.rollback()

                app.logger.exception(
                    "AI playground agent error."
                )

                response = (
                    "AI agent error. "
                    "Please try again."
                )

            if response:

                conversation = Conversation(

                    customer_id=customer.id,

                    message=message,

                    response=response
                )

                db.session.add(
                    conversation
                )

                try:

                    db.session.commit()

                except Exception:

                    db.session.rollback()

    if customer:

        conversations = (

            Conversation.query
            .filter_by(
                customer_id=customer.id
            )
            .order_by(
                Conversation.id.asc()
            )
            .all()
        )

    return render_template(

        "ai_playground.html",

        business=business,

        response=response,

        customer=customer,

        conversations=conversations,

        customer_phone=customer_phone,

        message=message,

        frontend_action=frontend_action
    )


# ============================================================
# AI ORDER CREATOR
# ============================================================

@app.route(
    "/business/<int:business_id>/ai-order",
    methods=["GET", "POST"]
)
def ai_order(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found."

    result = None

    phone = ""

    message = ""

    if request.method == "POST":

        action = request.form.get(
            "action",
            "generate"
        )

        # ====================================================
        # GENERATE ORDER
        # ====================================================

        if action == "generate":

            phone = request.form.get(
                "phone",
                ""
            ).strip()

            message = request.form.get(
                "message",
                ""
            ).strip()

            if not phone or not message:

                result = {
                    "error":
                        "Please enter the customer phone and message."
                }

            else:

                try:

                    extracted = extract_order(

                        business.id,

                        message
                    )

                    if not extracted.get(
                        "items"
                    ):

                        result = {

                            "error": (
                                "I couldn't find any available "
                                "menu items in that request."
                            )
                        }

                    else:

                        result = {

                            "order":
                                extracted,

                            "total":
                                extracted["total"],

                            "restaurant":
                                business.name,

                            "phone":
                                phone,

                            "message":
                                message,

                            "confirmed":
                                False
                        }

                        session[
                            "pending_ai_order"
                        ] = {

                            "business_id":
                                business.id,

                            "phone":
                                phone,

                            "message":
                                message,

                            "items":
                                extracted["items"],

                            "total":
                                extracted["total"]
                        }

                        session.modified = True

                except Exception as e:

                    result = {

                        "error":
                            f"Unable to generate order: {str(e)}"
                    }

        # ====================================================
        # CONFIRM ORDER
        # ====================================================

        elif action == "confirm":

            pending = session.get(
                "pending_ai_order"
            )

            if not pending:

                result = {

                    "error":
                        "No pending order to confirm."
                }

            elif (
                pending.get(
                    "business_id"
                )
                != business.id
            ):

                result = {

                    "error":
                        "This order does not belong to this business."
                }

            else:

                try:

                    phone = pending[
                        "phone"
                    ]

                    customer = (
                        Customer.query
                        .filter_by(
                            phone=phone,
                            business_id=business.id
                        )
                        .first()
                    )

                    if not customer:

                        customer = Customer(

                            name="New Customer",

                            phone=phone,

                            language="English",

                            business_id=business.id
                        )

                        db.session.add(
                            customer
                        )

                        db.session.flush()

                    # ========================================
                    # RE-CALCULATE FROM REAL MENU
                    # ========================================

                    final_items = []

                    final_total = 0

                    for item in pending[
                        "items"
                    ]:

                        menu = Menu.query.filter(

                            Menu.business_id
                            == business.id,

                            Menu.name
                            == item["name"],

                            Menu.available
                            == True

                        ).first()

                        if not menu:

                            continue

                        quantity = int(
                            item["quantity"]
                        )

                        subtotal = (

                            float(
                                menu.price
                            )
                            *
                            quantity
                        )

                        final_items.append({

                            "name":
                                menu.name,

                            "quantity":
                                quantity,

                            "price":
                                float(
                                    menu.price
                                ),

                            "subtotal":
                                subtotal
                        })

                        final_total += subtotal

                    if not final_items:

                        db.session.rollback()

                        result = {

                            "error": (
                                "The menu items in this order "
                                "are no longer available."
                            )
                        }

                    else:

                        # ====================================
                        # CREATE ORDER
                        # ====================================

                        order = Order(

                            customer_name=
                                customer.name,

                            customer_phone=
                                customer.phone,

                            delivery_address=
                                "Unknown",

                            total_price=
                                final_total,

                            status=
                                "Pending",

                            # IMPORTANT:
                            # Customer confirmation is NOT
                            # payment.
                            payment_status=
                                PAYMENT_UNPAID,

                            paid_at=
                                None,

                            payment_method=
                                None,

                            payment_transaction_id=
                                None,

                            business_id=
                                business.id,

                            customer_id=
                                customer.id
                        )

                        db.session.add(
                            order
                        )

                        db.session.flush()

                        # ====================================
                        # CREATE ORDER ITEMS
                        # ====================================

                        for item in final_items:

                            order_item = OrderItem(

                                name=
                                    item["name"],

                                quantity=
                                    item["quantity"],

                                price=
                                    item["price"],

                                subtotal=
                                    item["subtotal"],

                                order_id=
                                    order.id
                            )

                            db.session.add(
                                order_item
                            )

                        # ====================================
                        # RESERVE INVENTORY
                        # ====================================

                        inventory_result = (
                            reserve_inventory_for_order(order)
                        )

                        if not inventory_result["success"]:

                            db.session.rollback()

                            result = {
                                "error": (
                                    "This order cannot be confirmed "
                                    "because one or more ingredients "
                                    "are no longer available."
                                )
                            }

                        else:

                            db.session.commit()

                            session.pop(
                                "pending_ai_order",
                            None
                            )

                            result = {

                                "confirmed":
                                    True,

                                "order_id":
                                    order.id,

                                "restaurant":
                                    business.name,

                                "phone":
                                    customer.phone,

                                "items":
                                    final_items,

                                "total":
                                    final_total,

                                "payment_status":
                                    PAYMENT_UNPAID
                            }

                except Exception as e:

                    db.session.rollback()

                    result = {

                        "error":
                            (
                                "Unable to confirm order: "
                                f"{str(e)}"
                            )
                    }

    pending = session.get(
        "pending_ai_order"
    )

    if pending and not result:

        result = {

            "order": {

                "items":
                    pending["items"],

                "total":
                    pending["total"]
            },

            "total":
                pending["total"],

            "restaurant":
                business.name,

            "phone":
                pending["phone"],

            "message":
                pending["message"],

            "confirmed":
                False
        }

        phone = pending[
            "phone"
        ]

        message = pending[
            "message"
        ]

    return render_template(

        "ai_order.html",

        business=business,

        result=result,

        phone=phone,

        message=message
    )


# ============================================================
# AI ORDER MODIFICATION
# ============================================================

@app.route(
    "/business/<int:business_id>/ai-order/modify",
    methods=["POST"]
)
def ai_order_modify(
    business_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return "Business not found", 404

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    message = request.form.get(
        "message",
        ""
    ).strip()

    if not phone or not message:

        return {

            "success": False,

            "message": (
                "Customer phone and message "
                "are required."
            )

        }, 400

    customer = Customer.query.filter_by(

        phone=phone,

        business_id=business.id

    ).first()

    if not customer:

        return {

            "success": False,

            "message":
                "Customer not found."

        }, 404

    order = (

        Order.query

        .filter(

            Order.customer_id
            == customer.id,

            Order.business_id
            == business.id,

            Order.status.in_([
                "Pending",
                "Preparing"
            ])
        )

        .order_by(
            Order.id.desc()
        )

        .first()
    )

    if not order:

        return {

            "success": False,

            "message": (
                "No active order was found "
                "for this customer."
            )

        }, 404

    try:

        command = interpret_order_request(
            message,
            order
        )

    except Exception as e:

        return {

            "success": False,

            "message": (
                f"AI interpretation failed: {str(e)}"
            )

        }, 500

    if command.get(
        "action"
    ) == "unknown":

        return {

            "success": False,

            "message": (
                "I couldn't understand the "
                "requested order change."
            ),

            "command":
                command

        }, 400

    try:

        result = execute_order_action(

            business.id,

            order,

            command
        )

        result[
            "order_id"
        ] = order.id

        return result

    except Exception as e:

        db.session.rollback()

        return {

            "success": False,

            "message": (
                f"Unable to modify order: {str(e)}"
            )

        }, 500


# ============================================================
# UNIFIED AI AGENT API
# ============================================================

@app.route(
    "/api/agent/chat",
    methods=["POST"]
)
def agent_chat():

    if "user_id" not in session:

        return {

            "success": False,

            "error":
                "Unauthorized"

        }, 401

    data = request.get_json(
        silent=True
    )

    if not isinstance(
        data,
        dict
    ):

        return {

            "success": False,

            "error":
                "Invalid JSON request."

        }, 400

    business_id = data.get(
        "business_id"
    )

    phone = str(
        data.get(
            "phone",
            ""
        )
    ).strip()

    message = str(
        data.get(
            "message",
            ""
        )
    ).strip()

    if not business_id:

        return {

            "success": False,

            "error":
                "business_id is required."

        }, 400

    if not phone:

        return {

            "success": False,

            "error":
                "phone is required."

        }, 400

    if not message:

        return {

            "success": False,

            "error":
                "message is required."

        }, 400

    try:

        business_id = int(
            business_id
        )

    except (
        TypeError,
        ValueError
    ):

        return {

            "success": False,

            "error":
                "Invalid business_id."

        }, 400

    business = Business.query.filter_by(

        id=business_id,

        owner_id=session["user_id"]

    ).first()

    if not business:

        return {

            "success": False,

            "error":
                "Business not found."

        }, 404

    customer = Customer.query.filter_by(

        business_id=business.id,

        phone=phone

    ).first()

    if not customer:

        customer = Customer(

            name="New Customer",

            phone=phone,

            language="English",

            business_id=business.id
        )

        db.session.add(
            customer
        )

        db.session.commit()

    # ========================================================
    # DETECT CUSTOMER LANGUAGE FROM CURRENT MESSAGE
    # ========================================================

    detected_language = detect_customer_language(
        message,
        fallback=customer.language or "English"
    )

    language = detected_language

    if customer.language != detected_language:

        customer.language = detected_language

        db.session.commit()

    previous_conversations = (

        Conversation.query
        .filter_by(
            customer_id=customer.id
        )
        .order_by(
            Conversation.id.desc()
        )
        .limit(12)
        .all()
    )

    history = [

        {
            "message":
                chat.message,

            "response":
                chat.response
        }

        for chat in reversed(
            previous_conversations
        )
    ]

    try:

        result = run_agent(

            business.id,

            customer.phone,

            message,

            language,

            history,

            customer=customer,
        )

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Unified AI agent failed."
        )

        return {

            "success": False,

            "error": (
                "The AI agent could not "
                "process the request."
            )

        }, 500

    if not isinstance(
        result,
        dict
    ):

        return {

            "success": False,

            "error":
                "Invalid agent response."

        }, 500

    response_message = result.get(
        "message"
    )

    if response_message:

        conversation = Conversation(

            customer_id=customer.id,

            message=message,

            response=response_message
        )

        db.session.add(
            conversation
        )

        db.session.commit()

        return {
            "success": True,
            "type": result.get(
                "type",
                "response"
            ),
            "message": response_message,
            "action": result.get("action"),
            "business_id": business.id,
            "customer": {
                "id": customer.id,
                "name": customer.name,
                "phone": customer.phone,
                "language": customer.language
            }
        }


def _mark_whatsapp_message_completed(message_id):
    """Mark an inbound WhatsApp message as successfully processed."""
    if not message_id:
        return

    try:
        record = WhatsAppMessage.query.filter_by(
            message_id=message_id
        ).first()

        if record:
            record.mark_completed()
            db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Failed to mark WhatsApp message %s as completed",
            message_id,
        )


def _mark_whatsapp_message_failed(message_id, error):
    """Mark an inbound WhatsApp message as failed so a retry can reclaim it."""
    if not message_id:
        return

    try:
        record = WhatsAppMessage.query.filter_by(
            message_id=message_id
        ).first()

        if record:
            record.mark_failed(error)
            db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception(
            "Failed to mark WhatsApp message %s as failed",
            message_id,
        )


def process_whatsapp_image_async(
    business_id,
    from_phone,
    media_id,
    caption,
    contact_name,
    message_id,
):
    """
    Download and analyze a WhatsApp image, then pass the
    result into the central restaurant agent.
    """

    with app.app_context():

        try:

            app.logger.warning(
                "[ASYNC IMAGE] Starting image processing for %s",
                from_phone
            )

            image_bytes, mime_type = (
                download_whatsapp_image(
                    media_id
                )
            )

            import base64

            image_data_url = (
                "data:"
                f"{mime_type};base64,"
                f"{base64.b64encode(image_bytes).decode('utf-8')}"
            )

            vision_prompt = """
You are analyzing a photo sent to a restaurant AI assistant.

Return concise visual context that can help the restaurant assistant
understand why the customer may have sent the image.

Focus only on clearly visible information:
- Is this food or a meal?
- Is it a restaurant dish, packaged order, drink, or another
  restaurant-related image?
- Describe broad visible characteristics only.
- Mention an obvious visible quality/problem only when it is
  genuinely visible.

Do NOT:
- guess the exact dish name unless it is clearly identifiable;
- guess ingredients that cannot be clearly seen;
- guess taste, freshness, temperature, or safety;
- guess prices;
- guess the customer's feelings;
- claim something is wrong unless it is visually evident.

Return only concise visual context.
"""

            provider = OpenAIProvider()

            image_context = provider.analyze_image(
                image_data_url,
                vision_prompt,
                max_tokens=180,
            )

            customer_message = (
                caption.strip()
                if caption
                and caption.strip()
                else "Customer sent a photo."
            )

            app.logger.warning(
                "[ASYNC IMAGE] Caption: %s",
                customer_message
            )

            app.logger.warning(
                "[ASYNC IMAGE] Vision context: %s",
                image_context
            )

            process_whatsapp_message_async(
                business_id,
                from_phone,
                customer_message,
                contact_name,
                message_id,
                image_context=image_context,
            )

        except Exception as error:
            _mark_whatsapp_message_failed(message_id, error)

            db.session.rollback()

            app.logger.exception(
                "[ASYNC IMAGE] Failed to process image "
                "from %s",
                from_phone
            )

            send_whatsapp_message(
                from_phone,
                "Sorry, I couldn't process that image right now. "
                "Please try again."
            )

        finally:

            with WHATSAPP_IN_FLIGHT_LOCK:

                WHATSAPP_IN_FLIGHT.discard(
                    message_id
                )


def process_whatsapp_audio_async(
    business_id,
    from_phone,
    media_id,
    contact_name,
    message_id,
):

    with app.app_context():

        try:

            app.logger.warning(
                "[ASYNC AUDIO] Starting transcription for %s",
                from_phone
            )

            text_body = process_whatsapp_audio(
                media_id
            )

            if not text_body:

                raise RuntimeError(
                    "WhatsApp audio transcription returned empty text."
                )

            app.logger.warning(
                "[ASYNC AUDIO] Transcript: %s",
                text_body
            )

            process_whatsapp_message_async(
                business_id,
                from_phone,
                text_body,
                contact_name,
                message_id
            )

        except Exception as error:
            _mark_whatsapp_message_failed(message_id, error)

            db.session.rollback()

            app.logger.exception(
                "[ASYNC AUDIO] Failed to process voice message "
                "from %s",
                from_phone
            )

            send_whatsapp_message(
                from_phone,
                "Sorry, I couldn't understand that voice message. "
                "Please try again."
            )

        finally:

            with WHATSAPP_IN_FLIGHT_LOCK:

                WHATSAPP_IN_FLIGHT.discard(
                    message_id
                )

# ============================================================
# FAST WHATSAPP RESPONSE PATH
# ============================================================

def fast_whatsapp_response(
    business_id,
    phone,
    message,
):
    """
    Handle deterministic, high-frequency WhatsApp messages without
    loading conversation history or running the full AI agent.

    Returns:
        dict | None
    """

    from services.ai.agent import (
        CUSTOMER_LANGUAGE,
        clean_text,
        normalize_text,
        detect_customer_language,
        customer_response,
        classify_message_fast,
        update_pending_order_quantity,
        modify_pending_order,
    )

    message = clean_text(message)

    if not message:
        return None

    normalized = normalize_text(message)

    # --------------------------------------------------------
    # FAST LANGUAGE DETECTION
    # --------------------------------------------------------

    language = detect_customer_language(
        message,
        fallback="English",
    )

    CUSTOMER_LANGUAGE.set(language)

    # --------------------------------------------------------
    # FAST GREETINGS
    # --------------------------------------------------------

    greeting_responses = {
        "English": {
            "hi": "Hi! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "hey": "Hey! How can I help you today?",
            "good morning": "Good morning! How can I help you today?",
            "good afternoon": "Good afternoon! How can I help you today?",
            "good evening": "Good evening! How can I help you today?",
        },
        "French": {
            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "bonsoir": "Bonsoir ! Comment puis-je vous aider ?",
            "salut": "Salut ! Comment puis-je vous aider ?",
        },
        "Spanish": {
            "hola": "¡Hola! ¿Cómo puedo ayudarte?",
        },
        "Portuguese": {
            "olá": "Olá! Como posso ajudá-lo?",
        },
        "Italian": {
            "ciao": "Ciao! Come posso aiutarti?",
        },
        "German": {
            "hallo": "Hallo! Wie kann ich Ihnen helfen?",
        },
    }

    response = (
        greeting_responses
        .get(language, greeting_responses["English"])
        .get(normalized)
    )

    if response:
        return {
            "type": "response",
            "message": customer_response(
                response,
                message,
            ),
        }

    # --------------------------------------------------------
    # FAST QUANTITY-CHANGE DETECTION
    # --------------------------------------------------------
    #
    # Only call the database-backed quantity handler when the
    # message actually looks like a quantity update.

    quantity_pattern = re.compile(
        r"^(?:"
        r"make(?:\s+it|\s+that|\s+the)?|"
        r"change(?:\s+it|\s+the)?(?:\s+to)?|"
        r"set(?:\s+the)?|"
        r"actually\s+make(?:\s+it|\s+that|\s+the)?|"
        r"actually\s+change(?:\s+it|\s+the)?(?:\s+to)?|"
        r"actually\s+set(?:\s+the)?|"
        r"mets(?:\s+en)?|"
        r"met"
        r")\s+.+\s+"
        r"(?:\d+|one|two|three|four|five)"
        r"(?:\s+please)?$"
    )

    quantity_only_pattern = re.compile(
        r"^(?:"
        r"make\s+(?:it|that)|"
        r"change\s+it(?:\s+to)?|"
        r"set"
        r")\s+"
        r"(?:\d+|one|two|three|four|five)"
        r"(?:\s+please)?$"
    )

    looks_like_quantity_update = (
        bool(quantity_pattern.fullmatch(normalized))
        or bool(quantity_only_pattern.fullmatch(normalized))
    )

    if looks_like_quantity_update:

        result = update_pending_order_quantity(
            business_id,
            phone,
            message,
        )

        if result:
            return result

    # --------------------------------------------------------
    # FAST PENDING-ORDER MODIFICATION
    # --------------------------------------------------------
    #
    # "add", "remove", and similar modifications are already
    # classified locally. Try the existing pending-order handler
    # without invoking Groq.

    classification = classify_message_fast(
        message
    )

    if classification == "modify_order":

        result = modify_pending_order(
            business_id,
            phone,
            message,
            language,
        )

        if result is not None:
            return result

    return None


def save_fast_whatsapp_conversation(
    business_id,
    phone,
    message,
    response_message,
    customer_name=None,
):
    """
    Persist a fast-path WhatsApp conversation after the response
    has already been sent to the customer.
    """

    try:

        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business_id,
        ).first()

        if not customer:

            customer = Customer(
                name=customer_name or "New Customer",
                phone=phone,
                language="English",
                business_id=business_id,
            )

            db.session.add(customer)
            db.session.flush()

        elif (
            customer_name
            and (
                not customer.name
                or customer.name == "New Customer"
            )
        ):

            customer.name = customer_name

        conversation = Conversation(
            customer_id=customer.id,
            message=message,
            response=response_message,
        )

        db.session.add(conversation)
        db.session.commit()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "[FAST] Conversation persistence failed."
        )


# ============================================================
# WHATSAPP WEBHOOK
# ============================================================

def process_whatsapp_message_async(
    business_id,
    from_phone,
    text_body,
    contact_name,
    message_id,
    image_context=None,
):

    with app.app_context():
        whatsapp_processing_succeeded = False

        app.logger.warning(
            "[ASYNC] Worker started for %s: %s",
            from_phone,
            text_body,
        )

        try:

            # The webhook already resolved the business before
            # submitting this worker. Avoid another PostgreSQL query
            # on the critical WhatsApp response path.
            business = type(
                "WhatsAppBusinessRef",
                (),
                {"id": business_id},
            )()

            # ====================================================
            # FAST DETERMINISTIC WHATSAPP PATH
            # ====================================================

            fast_start = time.perf_counter()

            fast_result = fast_whatsapp_response(
                business_id,
                from_phone,
                text_body,
            )

            fast_elapsed = time.perf_counter() - fast_start

            if fast_result is not None:

                reply_text = (
                    fast_result.get("message")
                    or fast_result.get("response")
                    or ""
                )

                app.logger.warning(
                    "[PERF FAST] handler: %.3fs",
                    fast_elapsed,
                )

                send_start = time.perf_counter()

                send_whatsapp_message(
                    from_phone,
                    reply_text,
                )

                app.logger.warning(
                    "[PERF FAST] WhatsApp send: %.3fs",
                    time.perf_counter() - send_start,
                )

                # Persist after the customer has already received
                # the response, so database writes are not on the
                # critical response path.
                save_start = time.perf_counter()

                save_fast_whatsapp_conversation(
                    business_id=business_id,
                    phone=from_phone,
                    message=text_body,
                    response_message=reply_text,
                    customer_name=contact_name,
                )

                app.logger.warning(
                    "[PERF FAST] conversation save: %.3fs",
                    time.perf_counter() - save_start,
                )

                whatsapp_processing_succeeded = True
                return

            # ====================================================
            # FULL AI PATH
            # ====================================================

            app.logger.warning(
                "[ASYNC] Starting run_customer_agent"
            )

            ai_start = time.perf_counter()

            (
                agent_result,
                reply_text,
                customer
            ) = run_customer_agent(
                business,
                from_phone,
                text_body,
                contact_name,
                image_context=image_context,
            )

            app.logger.info(
                "[PERF] AI processing: %.2fs",
                time.perf_counter() - ai_start
            )

            send_start = time.perf_counter()

            send_whatsapp_message(
                from_phone,
                reply_text
            )

            app.logger.info(
                "[PERF] WhatsApp send: %.2fs",
                time.perf_counter() - send_start
            )

        except Exception as error:
            db.session.rollback()

            _mark_whatsapp_message_failed(
                message_id,
                error,
            )

            app.logger.exception(
                "Async WhatsApp processing failed for %s",
                from_phone
            )

        else:
            whatsapp_processing_succeeded = True

        finally:
            if whatsapp_processing_succeeded:
                _mark_whatsapp_message_completed(
                    message_id
                )

            with WHATSAPP_IN_FLIGHT_LOCK:
                WHATSAPP_IN_FLIGHT.discard(
                    message_id
                )

@app.route(
    "/webhook/whatsapp",
    methods=["GET", "POST"]
)
def whatsapp_webhook():

    import time
    webhook_start = time.perf_counter()

    if request.method == "GET":

        mode = request.args.get(
            "hub.mode"
        )

        token = request.args.get(
            "hub.verify_token"
        )

        challenge = request.args.get(
            "hub.challenge"
        )

        if (
            mode == "subscribe"
            and token == WHATSAPP_VERIFY_TOKEN
        ):

            return challenge, 200

        return (
            "Verification failed",
            403
        )

    data = request.get_json(
        silent=True
    )

    app.logger.warning(
        "WHATSAPP WEBHOOK PAYLOAD: %s",
        data
    )

    if not isinstance(
        data,
        dict
    ):

        return "OK", 200

    try:

        entries = data.get(
            "entry",
            []
        )

        if not isinstance(
            entries,
            list
        ):

            entries = []

        for entry in entries:

            if not isinstance(
                entry,
                dict
            ):

                continue

            changes = entry.get(
                "changes",
                []
            )

            if not isinstance(
                changes,
                list
            ):

                continue

            for change in changes:

                if not isinstance(
                    change,
                    dict
                ):

                    continue

                value = change.get(
                    "value",
                    {}
                )

                if not isinstance(
                    value,
                    dict
                ):

                    continue

                messages = value.get(
                    "messages",
                    []
                )

                if (
                    not isinstance(
                        messages,
                        list
                    )
                    or not messages
                ):

                    continue

                contacts = value.get(
                    "contacts",
                    []
                )

                if not isinstance(
                    contacts,
                    list
                ):

                    contacts = []

                business = find_whatsapp_business(
                    value
                )

                if not business:

                    continue

                for wa_message in messages:

                    contact_name = "New Customer"

                    if not isinstance(
                        wa_message,
                        dict
                    ):
                        continue

                    message_id = (
                        wa_message.get(
                            "id"
                        )
                    )

                    if not message_id:

                        app.logger.warning(
                            "WhatsApp message has no message ID."
                        )

                        continue

                    # ------------------------------------------------
                    # DURABLE WHATSAPP IDEMPOTENCY
                    # ------------------------------------------------
                    #
                    # The in-memory set protects against duplicates inside
                    # this Python process. The database record protects
                    # against duplicates across workers, restarts and
                    # deployments.
                    #
                    # A unique message_id is the final authority.

                    durable_message = WhatsAppMessage.query.filter_by(
                        message_id=message_id
                    ).first()

                    if durable_message:

                        # A completed message must never be processed again.
                        if durable_message.status == "completed":

                            app.logger.info(
                                "Skipping completed duplicate WhatsApp "
                                "message: %s",
                                message_id,
                            )

                            continue

                        # A processing record may belong to another live
                        # worker. Only reclaim it when it is stale.
                        if durable_message.status == "processing":

                            now = datetime.now(timezone.utc)
                            updated_at = durable_message.updated_at

                            if updated_at is not None and updated_at.tzinfo is None:
                                updated_at = updated_at.replace(
                                    tzinfo=timezone.utc
                                )

                            processing_age = (
                                now - updated_at
                                if updated_at is not None
                                else timedelta.max
                            )

                            if processing_age <= timedelta(minutes=10):

                                app.logger.info(
                                    "Skipping in-progress duplicate WhatsApp "
                                    "message: %s",
                                    message_id,
                                )

                                continue

                            app.logger.warning(
                                "Reclaiming stale WhatsApp message: %s "
                                "(age=%s)",
                                message_id,
                                processing_age,
                            )

                            durable_message.status = "processing"
                            durable_message.attempts = (
                                durable_message.attempts or 0
                            ) + 1
                            durable_message.updated_at = now
                            durable_message.error = None

                            db.session.commit()

                            app.logger.info(
                                "Stale WhatsApp message reclaimed: %s",
                                message_id,
                            )

                            # Continue through the normal dispatch path.

                        # Failed messages are allowed to retry.
                        durable_message.status = "processing"
                        durable_message.attempts = (
                            durable_message.attempts or 0
                        ) + 1
                        durable_message.updated_at = (
                            datetime.now(timezone.utc)
                        )
                        durable_message.error = None
                        db.session.commit()

                    else:

                        durable_message = WhatsAppMessage(
                            message_id=message_id,
                            business_id=business.id,
                            from_phone=wa_message.get("from"),
                            status="processing",
                            attempts=1,
                        )

                        db.session.add(durable_message)

                        try:

                            db.session.commit()

                        except IntegrityError:

                            db.session.rollback()

                            app.logger.info(
                                "Skipping concurrently claimed WhatsApp "
                                "message: %s",
                                message_id,
                            )

                            continue

                    with WHATSAPP_IN_FLIGHT_LOCK:

                        if message_id in WHATSAPP_IN_FLIGHT:

                            app.logger.info(
                                "Skipping duplicate WhatsApp "
                                "message: %s",
                                message_id
                            )

                            continue

                        WHATSAPP_IN_FLIGHT.add(
                            message_id
                        )

                    from_phone = (
                        wa_message.get(
                            "from"
                        )
                    )

                    message_type = (
                        wa_message.get(
                            "type"
                        )
                    )

                    # ------------------------------------------------
                    # TEXT MESSAGE
                    # ------------------------------------------------

                    if message_type == "text":

                        text_body = (
                            wa_message.get(
                                "text",
                                {}
                            )
                            or {}
                        ).get(
                            "body",
                            ""
                        ).strip()

                    # ------------------------------------------------
                    # VOICE MESSAGE
                    # ------------------------------------------------

                    elif message_type == "audio":

                        audio_data = (
                            wa_message.get(
                                "audio",
                                {}
                            )
                            or {}
                        )

                        media_id = audio_data.get(
                            "id"
                        )

                        if not media_id:

                            app.logger.warning(
                                "WhatsApp audio message "
                                "has no media ID."
                            )

                            _mark_whatsapp_message_failed(
                                message_id,
                                "Audio message has no media ID.",
                            )

                            with WHATSAPP_IN_FLIGHT_LOCK:

                                WHATSAPP_IN_FLIGHT.discard(
                                    message_id
                                )

                            continue

                        try:

                            WHATSAPP_EXECUTOR.submit(
                                process_whatsapp_audio_async,
                                business.id,
                                from_phone,
                                media_id,
                                contact_name,
                                message_id,
                            )

                        except Exception as exc:

                            _mark_whatsapp_message_failed(
                                message_id,
                                f"Failed to submit WhatsApp audio worker: {exc}",
                            )

                            with WHATSAPP_IN_FLIGHT_LOCK:

                                WHATSAPP_IN_FLIGHT.discard(
                                    message_id
                                )

                            app.logger.exception(
                                "Failed to submit WhatsApp audio worker "
                                "for message %s",
                                message_id,
                            )

                        continue


                        try:

                            text_body = (
                                process_whatsapp_audio(
                                    media_id
                                )
                            )

                        except Exception:

                            app.logger.exception(
                                "Failed to process WhatsApp "
                                "voice message from %s",
                                from_phone
                            )

                            send_whatsapp_message(
                                from_phone,
                                "Sorry, I couldn't understand "
                                "that voice message. Please "
                                "try again."
                            )

                            with WHATSAPP_IN_FLIGHT_LOCK:

                                WHATSAPP_IN_FLIGHT.discard(
                                    message_id
                                )

                            continue

                    # ------------------------------------------------
                    # IMAGE MESSAGE
                    # ------------------------------------------------

                    elif message_type == "image":

                        image_data = (
                            wa_message.get(
                                "image",
                                {}
                            )
                            or {}
                        )

                        media_id = image_data.get(
                            "id"
                        )

                        caption = (
                            image_data.get(
                                "caption",
                                ""
                            )
                            or ""
                        ).strip()

                        if not media_id:

                            app.logger.warning(
                                "WhatsApp image message "
                                "has no media ID."
                            )

                            _mark_whatsapp_message_failed(
                                message_id,
                                "Image message has no media ID.",
                            )

                            with WHATSAPP_IN_FLIGHT_LOCK:

                                WHATSAPP_IN_FLIGHT.discard(
                                    message_id
                                )

                            continue

                        try:

                            WHATSAPP_EXECUTOR.submit(
                                process_whatsapp_image_async,
                                business.id,
                                from_phone,
                                media_id,
                                caption,
                                contact_name,
                                message_id,
                            )

                        except Exception as exc:

                            _mark_whatsapp_message_failed(
                                message_id,
                                f"Failed to submit WhatsApp image worker: {exc}",
                            )

                            with WHATSAPP_IN_FLIGHT_LOCK:

                                WHATSAPP_IN_FLIGHT.discard(
                                    message_id
                                )

                            app.logger.exception(
                                "Failed to submit WhatsApp image worker "
                                "for message %s",
                                message_id,
                            )

                        continue

                    # ------------------------------------------------
                    # UNSUPPORTED MESSAGE TYPE
                    # ------------------------------------------------

                    else:

                        app.logger.info(
                            "Ignoring unsupported WhatsApp "
                            "message type: %s",
                            message_type
                        )

                        _mark_whatsapp_message_completed(
                            message_id
                        )

                        with WHATSAPP_IN_FLIGHT_LOCK:

                            WHATSAPP_IN_FLIGHT.discard(
                                message_id
                            )

                        continue

                    if not from_phone or not text_body:

                        _mark_whatsapp_message_failed(
                            message_id,
                            "WhatsApp text message is missing sender or body.",
                        )

                        with WHATSAPP_IN_FLIGHT_LOCK:

                            WHATSAPP_IN_FLIGHT.discard(
                                message_id
                            )

                        continue

                    # ------------------------------------------------
                    # SEND WHATSAPP MESSAGE TO N8N
                    # ------------------------------------------------

                    n8n_webhook_url = os.environ.get(
                        "N8N_WEBHOOK_URL"
                    )

                    # n8n is optional. Do not attempt localhost/127.0.0.1
                    # webhooks when the local n8n service is not running.
                    if (
                        n8n_webhook_url
                        and not n8n_webhook_url.startswith(
                            ("http://localhost:", "http://127.0.0.1:")
                        )
                    ):

                        WHATSAPP_EXECUTOR.submit(
                            send_n8n_webhook_async,
                            n8n_webhook_url,
                            message_id,
                            business.id,
                            from_phone,
                            message_type,
                            text_body,
                        )

                    contact_name = (
                        "New Customer"
                    )

                    for contact in contacts:

                        if not isinstance(
                            contact,
                            dict
                        ):

                            continue

                        if (
                            contact.get(
                                "wa_id"
                            )
                            == from_phone
                        ):

                            contact_name = (

                                contact.get(
                                    "profile",
                                    {}
                                )
                                or {}
                            ).get(
                                "name",
                                "New Customer"
                            )

                            try:

                                WHATSAPP_EXECUTOR.submit(
                                    process_whatsapp_message_async,
                                    business.id,
                                    from_phone,
                                    text_body,
                                    contact_name,
                                    message_id,
                                )

                            except Exception as exc:

                                _mark_whatsapp_message_failed(
                                    message_id,
                                    f"Failed to submit WhatsApp worker: {exc}",
                                )

                                with WHATSAPP_IN_FLIGHT_LOCK:

                                    WHATSAPP_IN_FLIGHT.discard(
                                        message_id
                                    )

                                app.logger.exception(
                                    "Failed to submit WhatsApp worker "
                                    "for message %s",
                                    message_id,
                                )

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "WhatsApp webhook processing failed."
        )

    return "OK", 200


# ============================================================
# CONTACT SUPPORT
# ============================================================

@app.route(
    "/support",
    methods=["GET", "POST"]
)
def support():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        subject = request.form.get(
            "subject",
            ""
        ).strip()

        message = request.form.get(
            "message",
            ""
        ).strip()

        if (
            not name
            or not email
            or not subject
            or not message
        ):

            flash(
                "Please complete every support field.",
                "error"
            )

            return render_template(
                "support.html"
            )

        if (
            "@"
            not in email
            or "."
            not in email.rsplit(
                "@",
                1
            )[-1]
        ):

            flash(
                "Please enter a valid email address.",
                "error"
            )

            return render_template(
                "support.html"
            )

        if len(subject) > 200:

            flash(
                "Subject is too long.",
                "error"
            )

            return render_template(
                "support.html"
            )

        ticket = SupportTicket(

            user_id=session.get(
                "user_id"
            ),

            name=name,

            email=email,

            subject=subject,

            message=message,

            status="Open",

            priority="Normal"
        )

        try:

            db.session.add(
                ticket
            )

            db.session.commit()

        except Exception:

            db.session.rollback()

            app.logger.exception(
                "Could not save support ticket."
            )

            flash(
                "We couldn't save your support request. Please try again.",
                "error"
            )

            return render_template(
                "support.html"
            )

        email_sent = False

        if SUPPORT_EMAIL:

            email_sent = send_support_email(

                SUPPORT_EMAIL,

                f"[Botify Support #{ticket.id}] "
                f"{ticket.subject}",

                (
                    f"New Botify support ticket "
                    f"#{ticket.id}\n\n"

                    f"Name: {ticket.name}\n"

                    f"Email: {ticket.email}\n"

                    f"Priority: {ticket.priority}\n"

                    f"Status: {ticket.status}\n\n"

                    f"Message:\n{ticket.message}\n"
                ),

                reply_to=ticket.email
            )

        if email_sent:

            send_support_email(

                ticket.email,

                f"Botify support request "
                f"#{ticket.id}",

                (
                    f"Hi {ticket.name},\n\n"

                    f"We received your support request: "
                    f"“{ticket.subject}”.\n\n"

                    f"Your ticket number is "
                    f"#{ticket.id}. "
                    f"Our team will review it and "
                    f"get back to you.\n\n"

                    f"Botify Support"
                )
            )

        if email_sent:

            flash(

                f"Ticket #{ticket.id} created. "
                f"A confirmation was sent to your email.",

                "success"
            )

        else:

            flash(

                f"Ticket #{ticket.id} created successfully. "
                f"Email notifications are not configured yet.",

                "success"
            )

        return redirect(
            url_for("support")
        )

    return render_template(
        "support.html"
    )


# ============================================================
# MY SUPPORT TICKETS
# ============================================================

@app.route(
    "/support/tickets"
)
def my_support_tickets():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    tickets = (

        SupportTicket.query
        .filter_by(
            user_id=session["user_id"]
        )
        .order_by(
            SupportTicket.created_at.desc()
        )
        .all()
    )

    return render_template(

        "support_tickets.html",

        tickets=tickets
    )


# ============================================================
# ADMIN SUPPORT
# ============================================================

def support_admin_required():

    user_id = session.get(
        "user_id"
    )

    if not user_id:

        return None, redirect(
            url_for("login")
        )

    user = db.session.get(
        User,
        user_id
    )

    if (
        not user
        or not ADMIN_EMAIL
        or user.email.lower()
        != ADMIN_EMAIL.lower()
    ):

        return None, (
            "Unauthorized",
            403
        )

    return user, None


@app.route(
    "/admin/support"
)
def admin_support():

    user, error = support_admin_required()

    if error:

        return error

    status = request.args.get(
        "status",
        ""
    ).strip()

    query = SupportTicket.query

    if status:

        query = query.filter_by(
            status=status
        )

    tickets = (

        query
        .order_by(
            SupportTicket.created_at.desc()
        )
        .all()
    )

    return render_template(

        "admin_support.html",

        tickets=tickets,

        current_status=status
    )


@app.route(
    "/admin/support/<int:ticket_id>/status",
    methods=["POST"]
)
def update_support_ticket_status(
    ticket_id
):

    user, error = support_admin_required()

    if error:

        return error

    ticket = db.session.get(
        SupportTicket,
        ticket_id
    )

    if not ticket:

        return (
            "Support ticket not found.",
            404
        )

    allowed_statuses = {

        "Open",

        "In Progress",

        "Resolved"
    }

    new_status = request.form.get(
        "status",
        ""
    ).strip()

    if new_status not in allowed_statuses:

        flash(
            "Invalid support ticket status.",
            "error"
        )

        return redirect(
            url_for(
                "admin_support"
            )
        )

    ticket.status = new_status

    try:

        db.session.commit()

    except Exception:

        db.session.rollback()

        flash(
            "Could not update the ticket.",
            "error"
        )

        return redirect(
            url_for(
                "admin_support"
            )
        )

    if new_status in {
        "In Progress",
        "Resolved"
    }:

        send_support_email(

            ticket.email,

            f"Botify support ticket "
            f"#{ticket.id} updated",

            (
                f"Hi {ticket.name},\n\n"

                f"Your Botify support ticket "
                f"#{ticket.id} is now marked as: "
                f"{new_status}.\n\n"

                f"Subject: {ticket.subject}\n\n"

                f"Botify Support"
            )
        )

    flash(
        f"Ticket #{ticket.id} marked as {new_status}.",
        "success"
    )

    return redirect(
        url_for("admin_support")
    )

# ============================================================
# PAYDUNYA PAYMENT CONFIG
# ============================================================

PAYDUNYA_MASTER_KEY = os.environ.get(
    "PAYDUNYA_MASTER_KEY",
    ""
)

PAYDUNYA_PRIVATE_KEY = os.environ.get(
    "PAYDUNYA_PRIVATE_KEY",
    ""
)

PAYDUNYA_TOKEN = os.environ.get(
    "PAYDUNYA_TOKEN",
    ""
)

PAYDUNYA_MODE = os.environ.get(
    "PAYDUNYA_MODE",
    "sandbox"
).lower()


if PAYDUNYA_MODE == "live":

    PAYDUNYA_BASE_URL = (
        "https://app.paydunya.com/api/v1"
    )

else:

    PAYDUNYA_BASE_URL = (
        "https://app.paydunya.com/sandbox-api/v1"
    )


PAYDUNYA_CREATE_URL = (
    f"{PAYDUNYA_BASE_URL}/checkout-invoice/create"
)

PAYDUNYA_CONFIRM_URL = (
    f"{PAYDUNYA_BASE_URL}/checkout-invoice/confirm"
)

PAYDUNYA_RETURN_URL = os.environ.get(
    "PAYDUNYA_RETURN_URL",
    ""
)

PAYDUNYA_CALLBACK_URL = os.environ.get(
    "PAYDUNYA_CALLBACK_URL",
    ""
)


# ============================================================
# PAYDUNYA HELPERS
# ============================================================

def paydunya_configured():

    return bool(
        PAYDUNYA_MASTER_KEY
        and PAYDUNYA_PRIVATE_KEY
        and PAYDUNYA_TOKEN
    )


def paydunya_headers():

    return {
        "Content-Type":
            "application/json",

        "PAYDUNYA-MASTER-KEY":
            PAYDUNYA_MASTER_KEY,

        "PAYDUNYA-PRIVATE-KEY":
            PAYDUNYA_PRIVATE_KEY,

        "PAYDUNYA-TOKEN":
            PAYDUNYA_TOKEN
    }


def create_paydunya_invoice(order):

    """
    Create a PayDunya checkout invoice.

    The order remains unpaid until PayDunya
    confirms that the payment was completed.
    """

    if not paydunya_configured():
        return {
            "success": False,
            "error": "PayDunya credentials are not configured."
        }

    if not order:
        return {
            "success": False,
            "error": "Order not found."
        }

    if order.status == "Cancelled":
        return {
            "success": False,
            "error": "Cancelled orders cannot be paid."
        }

    if is_order_paid(order):
        return {
            "success": False,
            "error": "This order has already been paid."
        }

    try:

        # ========================================================
        # SERIALIZE INVOICE CREATION PER ORDER
        # ========================================================
        #
        # The first lightweight checks above are only advisory.
        # The authoritative state check happens after acquiring
        # the PostgreSQL row lock below.
        #
        # Holding this lock until the PayDunya request and local
        # commit complete guarantees that concurrent workers cannot
        # both create invoices for the same order.

        locked_order = (
            Order.query
            .filter_by(id=order.id)
            .populate_existing()
            .with_for_update()
            .first()
        )

        if not locked_order:
            db.session.rollback()
            return {
                "success": False,
                "error": "Order not found."
            }

        order = locked_order

        # Re-check all mutable payment/order state AFTER the lock.
        if order.payment_status == "Paid":
            payment_record = Payment.query.filter_by(
                order_id=order.id
            ).first()

            return {
                "success": False,
                "error": "This order has already been paid.",
                "already_paid": True,
                "order_id": order.id,
                "payment_token": order.payment_token,
                "checkout_url": (
                    payment_record.checkout_url
                    if payment_record
                    else None
                )
            }

        if order.status in {
            "Cancelled",
            "Rejected",
            "Delivered",
            "Completed",
        }:
            return {
                "success": False,
                "error": (
                    "This order cannot receive a new payment "
                    f"because its status is {order.status}."
                ),
                "order_id": order.id,
                "status": order.status
            }

        # Another worker may have completed invoice creation while
        # this worker was waiting for the row lock.
        if order.payment_token:
            payment_record = Payment.query.filter_by(
                order_id=order.id
            ).first()

            if payment_record and payment_record.status == "Paid":
                return {
                    "success": False,
                    "error": "This order has already been paid.",
                    "already_paid": True,
                    "order_id": order.id,
                    "payment_token": order.payment_token,
                    "checkout_url": payment_record.checkout_url
                }

            return {
                "success": True,
                "token": str(order.payment_token),
                "checkout_url": (
                    payment_record.checkout_url
                    if payment_record
                    else None
                ),
                "already_exists": True
            }

        app.logger.warning(
            "========== PAYDUNYA CREATE FUNCTION REACHED =========="
        )

        import requests

        items = {}

        for index, item in enumerate(order.items):

            items[f"item_{index}"] = {
                "name": item.name,
                "quantity": int(item.quantity or 1),
                "unit_price": str(
                    int(float(item.price or 0))
                ),
                "total_price": str(
                    int(float(item.subtotal or 0))
                ),
                "description": ""
            }

        actions = {}

        if PAYDUNYA_RETURN_URL:
            actions["return_url"] = PAYDUNYA_RETURN_URL

        if PAYDUNYA_CALLBACK_URL:
            actions["callback_url"] = PAYDUNYA_CALLBACK_URL

        payload = {
            "invoice": {
                "items": items,
                "total_amount": int(
                    float(order.total_price or 0)
                ),
                "description": (
                    f"Order #{order.id} - "
                    f"{order.customer_name}"
                ),
                "customer": {
                    "name": (
                        order.customer_name
                        or "Customer"
                    ),
                    "phone": (
                        order.customer_phone
                        or ""
                    )
                }
            },

            "store": {
                "name": "Botify AI"
            },

            "custom_data": {
                "order_id": str(order.id),
                "business_id": str(order.business_id)
            }
        }

        if actions:
            payload["actions"] = actions

        app.logger.warning(
            "========== SENDING PAYDUNYA REQUEST =========="
        )

        app.logger.warning(
            "PayDunya URL: %s",
            PAYDUNYA_CREATE_URL
        )

        response = requests.post(
            PAYDUNYA_CREATE_URL,
            headers=paydunya_headers(),
            json=payload,
            timeout=20
        )

        app.logger.warning(
            "========== PAYDUNYA RESPONSE %s ==========",
            response.status_code
        )

        app.logger.warning(
            "PayDunya response: %s",
            response.text[:2000]
        )

        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:

            app.logger.error(
                "PayDunya HTTP error %s: %s",
                response.status_code,
                response.text[:1000]
            )

            return {
                "success": False,
                "error": "PayDunya request failed.",
                "details": data
            }

        response_code = str(
            data.get("response_code", "")
        )

        if response_code != "00":

            return {
                "success": False,
                "error": data.get(
                    "response_text",
                    "PayDunya could not create the payment."
                ),
                "details": data
            }

        token = data.get("token")

        checkout_url = (
            data.get("response_text")
            or data.get("checkout_url")
            or data.get("response_url")
        )

        if not token:

            return {
                "success": False,
                "error": (
                    "PayDunya did not return "
                    "a payment token."
                )
            }

        if not checkout_url:

            return {
                "success": False,
                "error": (
                    "PayDunya did not return "
                    "a checkout URL."
                )
            }

        if not hasattr(order, "payment_token"):

            return {
                "success": False,
                "error": (
                    "Order model is missing "
                    "payment_token."
                )
            }

        order.payment_token = str(token)

        payment_record = Payment.query.filter_by(
            order_id=order.id
        ).first()

        if not payment_record:

            payment_record = Payment(
                order_id=order.id,
                business_id=order.business_id,
                amount=float(
                    order.total_price or 0
                ),
                status="Pending",
                method="PayDunya"
            )

            db.session.add(payment_record)

        else:

            payment_record.amount = float(
                order.total_price or 0
            )

            payment_record.business_id = (
                order.business_id
            )

            if payment_record.status != "Paid":
                payment_record.status = "Pending"

            payment_record.method = "PayDunya"

        # Persist the checkout URL so concurrent requests that
        # discover the existing invoice can reuse the same URL.
        payment_record.checkout_url = checkout_url

        db.session.commit()

        return {
            "success": True,
            "token": str(token),
            "checkout_url": checkout_url
        }

    except Exception as e:

        db.session.rollback()

        app.logger.exception(
            "PAYDUNYA INVOICE CREATION EXCEPTION: %s",
            str(e)
        )

        return {
            "success": False,
            "error": f"PayDunya error: {str(e)}"
        }
      
def confirm_paydunya_payment(token):

    """
    Ask PayDunya for the authoritative status
    of a payment.
    """

    if not paydunya_configured():

        return {

            "success":
                False,

            "error":
                (
                    "PayDunya credentials are not configured."
                )
        }


    if not token:

        return {

            "success":
                False,

            "error":
                "Payment token is missing."
        }


    try:

        import requests


        response = requests.get(

            f"{PAYDUNYA_CONFIRM_URL}/{token}",

            headers=
                paydunya_headers(),

            timeout=20
        )


        try:

            data = response.json()

        except ValueError:

            data = {}


        if not response.ok:

            return {

                "success":
                    False,

                "error":
                    "PayDunya status request failed.",

                "details":
                    data
            }


        response_code = str(
            data.get(
                "response_code",
                ""
            )
        )


        if response_code != "00":

            return {

                "success":
                    False,

                "error":
                    data.get(

                        "response_text",

                        "Unable to verify payment."
                    ),

                "details":
                    data
            }


        invoice = data.get(
            "invoice",
            {}
        )


        if not isinstance(
            invoice,
            dict
        ):

            invoice = {}


        status = str(

            invoice.get(
                "status"
            )

            or data.get(
                "status"
            )

            or ""
        ).lower().strip()


        transaction_id = (

            invoice.get(
                "transaction_id"
            )

            or data.get(
                "transaction_id"
            )
        )


        payment_method = (

            invoice.get(
                "channel"
            )

            or invoice.get(
                "payment_method"
            )

            or data.get(
                "channel"
            )
        )


        return {

            "success":
                True,

            "status":
                status,

            "transaction_id":
                transaction_id,

            "payment_method":
                payment_method,

            "data":
                data
        }


    except Exception:

        app.logger.exception(

            "PayDunya payment verification failed."
        )

        return {

            "success":
                False,

            "error":
                (
                    "Could not verify payment "
                    "with PayDunya."
                )
        }


def process_paydunya_payment(token):

    """
    Verify a PayDunya payment and update the
    corresponding order.
    """

    if not token:
        return {
            "success": False,
            "error": "Missing payment token."
        }

    order = Order.query.filter_by(
        payment_token=str(token)
    ).first()

    if not order:
        return {
            "success": False,
            "error": (
                "No order was found for "
                "this payment."
            )
        }

    payment = confirm_paydunya_payment(
        str(token)
    )

    if not payment.get("success"):
        return payment

    status = (
        payment.get("status", "")
        or ""
    ).lower().strip()

    # ========================================================
    # SUCCESSFUL PAYMENT
    # ========================================================

    if status in {
        "completed",
        "complete",
        "paid",
        "success",
        "successful"
    }:

        if order.status == "Cancelled":
            return {
                "success": False,
                "paid": False,
                "error": (
                    "This order was cancelled "
                    "and cannot be marked paid."
                )
            }

        payment_method = (
            payment.get("payment_method")
            or "PayDunya"
        )

        transaction_id = payment.get(
            "transaction_id"
        )

        # ----------------------------------------------------
        # MARK ORDER AS PAID
        # ----------------------------------------------------

        mark_order_as_paid(
            order,
            payment_method=payment_method,
            transaction_id=transaction_id
        )

        # ----------------------------------------------------
        # MARK PAYMENT RECORD AS PAID
        # ----------------------------------------------------

        payment_record = Payment.query.filter_by(
            order_id=order.id
        ).first()

        if not payment_record:
            payment_record = Payment(
                order_id=order.id,
                business_id=order.business_id,
                amount=float(
                    order.total_price or 0
                ),
                status="Pending"
            )

            db.session.add(payment_record)

        payment_record.mark_paid(
            method=payment_method,
            transaction_id=transaction_id
        )

        db.session.commit()

        return {
            "success": True,
            "paid": True,
            "order_id": order.id,
            "status": "Completed"
        }

    # ========================================================
    # FAILED / CANCELLED
    # ========================================================

    if status in {
        "cancelled",
        "canceled",
        "failed",
        "declined"
    }:

        return {
            "success": True,
            "paid": False,
            "order_id": order.id,
            "status": status
        }

    # ========================================================
    # PENDING / UNKNOWN
    # ========================================================

    return {
        "success": True,
        "paid": False,
        "order_id": order.id,
        "status": status or "pending"
    }


# ============================================================
# PAYDUNYA: START PAYMENT
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/paydunya",
    methods=["GET", "POST"]
)
def start_paydunya_payment(
    business_id,
    order_id
):

    if "user_id" not in session:

        return redirect(
            "/login"
        )


    business = Business.query.filter_by(

        id=business_id,

        owner_id=session["user_id"]

    ).first()


    if not business:

        return (
            "Business not found.",
            404
        )


    order = Order.query.filter_by(

        id=order_id,

        business_id=business.id

    ).first()


    if not order:

        return (
            "Order not found.",
            404
        )


    result = create_paydunya_invoice(
        order
    )


    if not result.get(
        "success"
    ):

        flash(

            result.get(
                "error",
                "Could not start payment."
            ),

            "error"
        )

        return redirect(

            f"/business/"
            f"{business.id}/orders"
        )


    checkout_url = result.get(
        "checkout_url"
    )


    if not checkout_url:

        flash(

            "PayDunya did not return a checkout URL.",

            "error"
        )

        return redirect(

            f"/business/"
            f"{business.id}/orders"
        )


    return redirect(
        checkout_url
    )


# ============================================================
# PAYDUNYA: RETURN
# ============================================================

@app.route(
    "/payment/paydunya/return",
    methods=["GET"]
)
def paydunya_return():

    token = (
        request.args.get("token", "")
        or request.args.get("invoice_token", "")
    ).strip()

    if not token:
        return (
            "Payment token is missing.",
            400
        )

    result = process_paydunya_payment(token)

    order = Order.query.filter_by(
        payment_token=token
    ).first()

    if not order:
        return (
            "Order not found.",
            404
        )

    if result.get("paid"):
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Successful</title>
    <style>
        body {{
            margin: 0;
            padding: 40px 20px;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #1f2937;
            text-align: center;
        }}
        .card {{
            max-width: 480px;
            margin: 60px auto;
            padding: 40px 25px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.08);
        }}
        .success {{
            font-size: 56px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin-bottom: 10px;
        }}
        p {{
            color: #6b7280;
            line-height: 1.6;
        }}
        .order {{
            margin: 25px 0;
            padding: 15px;
            background: #f3f4f6;
            border-radius: 12px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="success">✓</div>
        <h1>Payment Successful</h1>
        <p>Your payment has been confirmed.</p>
        <div class="order">
            <strong>Order #{order.id}</strong>
        </div>
        <p>
            You can now return to WhatsApp.
            Your order is being processed.
        </p>
    </div>
</body>
</html>
"""

    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Payment Pending</title>
    <style>
        body {
            margin: 0;
            padding: 40px 20px;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #1f2937;
            text-align: center;
        }
        .card {
            max-width: 480px;
            margin: 60px auto;
            padding: 40px 25px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.08);
        }
        p {
            color: #6b7280;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Payment Pending</h1>
        <p>Your payment has not been confirmed yet.</p>
        <p>Please return to WhatsApp and check your order status.</p>
    </div>
</body>
</html>
"""


# ============================================================
# PAYDUNYA: IPN / CALLBACK
# ============================================================

@app.route(
    "/payment/paydunya/ipn",
    methods=["POST"]
)
@app.route(
    "/payment/paydunya/callback",
    methods=["POST"]
)
def paydunya_ipn():

    data = request.get_json(
        silent=True
    )


    if not isinstance(
        data,
        dict
    ):

        data = request.form.to_dict()


    token = (

        data.get(
            "token"
        )

        or data.get(
            "invoice_token"
        )
    )


    # Some callback structures may nest
    # the payment data.

    if not token:

        nested = data.get(
            "data"
        )


        if isinstance(
            nested,
            dict
        ):

            token = (

                nested.get(
                    "token"
                )

                or nested.get(
                    "invoice_token"
                )
            )


    if not token:

        return {

            "success":
                False,

            "error":
                "Missing payment token."
        }, 400


    result = process_paydunya_payment(
        str(token).strip()
    )


    if not result.get(
        "success"
    ):

        app.logger.warning(

            "PayDunya IPN failed: %s",

            result
        )

        return {

            "success":
                False,

            "error":
                result.get(
                    "error",
                    "Payment processing failed."
                )
        }, 400


    return {
        "success": True,
        "paid": result.get(
            "paid",
            False
        ),
        "status": result.get(
            "status"
        ),
        "order_id": result.get(
            "order_id"
        )
    }, 200


# ============================================================
# PAYDUNYA: PAYMENT STATUS
# ============================================================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/paydunya/status",
    methods=["GET"]
)
def paydunya_payment_status(
    business_id,
    order_id
):

    if "user_id" not in session:

        return {

            "success":
                False,

            "error":
                "Unauthorized"
        }, 401


    business = Business.query.filter_by(

        id=business_id,

        owner_id=session["user_id"]

    ).first()


    if not business:

        return {

            "success":
                False,

            "error":
                "Business not found."
        }, 404


    order = Order.query.filter_by(

        id=order_id,

        business_id=business.id

    ).first()


    if not order:

        return {

            "success":
                False,

            "error":
                "Order not found."
        }, 404


    token = getattr(

        order,

        "payment_token",

        None
    )


    if not token:

        return {

            "success":
                False,

            "error":
                (
                    "This order does not have "
                    "a PayDunya payment."
                )
        }, 400


    result = process_paydunya_payment(
        token
    )




    return {
        "success": result.get(
            "success",
            False
        ),
        "paid": result.get(
            "paid",
            False
        ),
        "status": result.get(
            "status"
        ),
        "payment_status": getattr(
            order,
            "payment_status",
            PAYMENT_UNPAID
        ),
        "order_id": order.id
    }

# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout"
)
def logout():

    session.pop(
        "user_id",
        None
    )

    return redirect(
        "/login"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route(
    "/about"
)
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


@app.errorhandler(500)
def server_error(error):

    return render_template(
        "500.html"
    ), 500


# ============================================================
# CONTEXT PROCESSOR
# ============================================================

@app.context_processor
def inject_globals():

    return {
        "app_name": "Botify AI",
        "currency": "FCFA",
        "payment_paid": PAYMENT_PAID,
        "payment_unpaid": PAYMENT_UNPAID
    }

# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                "5000"
            )
        ),

        debug=(
            os.environ.get(
                "FLASK_DEBUG",
                "false"
            ).lower()
            == "true"
        )
    )
