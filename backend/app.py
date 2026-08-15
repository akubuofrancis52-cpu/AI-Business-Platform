import os
from langdetect import detect, DetectorFactory

from datetime import datetime, timedelta

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
from models.conversation import Conversation
from models.pending_order import PendingOrder
from models.support_ticket import SupportTicket

from services.ai.prompt_builder import build_restaurant_prompt
from services.ai.openai_provider import OpenAIProvider

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

    if len(text) < 4:
        return fallback

    try:

        detected_code = detect(text)

        return LANGUAGE_MAP.get(
            detected_code,
            fallback
        )

    except Exception:

        app.logger.exception(
            "Customer language detection failed."
        )

        return fallback


app = Flask(__name__)


# ============================================================
# APPLICATION CONFIG
# ============================================================

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "dev-only-change-me"
)


app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///business.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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

    order.paid_at = datetime.utcnow()

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
    customer_name=None
):

    """
    Find/create customer, load recent memory, run central
    AI agent and save conversation.
    """

    customer = Customer.query.filter_by(
        phone=phone,
        business_id=business.id
    ).first()

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

    detected_language = detect_customer_language(
        message,
        fallback=customer.language or "English"
    )

    if detected_language != customer.language:

        customer.language = detected_language


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
            "message": chat.message,
            "response": chat.response
        }

        for chat in reversed(
            previous_conversations
        )
    ]

    result = run_agent(
        business.id,
        customer.phone,
        message,
        detected_language,
        history
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

    db.session.commit()

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

    today = datetime.utcnow().date()

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

    order.status = new_status

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

    language = (

        customer.language

        or "English"
    )

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

            history
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


# ============================================================
# WHATSAPP WEBHOOK
# ============================================================

@app.route(
    "/webhook/whatsapp",
    methods=["GET", "POST"]
)
def whatsapp_webhook():

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

                    if not isinstance(
                        wa_message,
                        dict
                    ):

                        continue

                    if (
                        wa_message.get(
                            "type"
                        )
                        != "text"
                    ):

                        continue

                    from_phone = (
                        wa_message.get(
                            "from"
                        )
                    )

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

                    if not from_phone or not text_body:

                        continue

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

                        (
                            agent_result,
                            reply_text,
                            customer
                        ) = run_customer_agent(

                            business,

                            from_phone,

                            text_body,

                            contact_name
                        )

                    except Exception:

                        db.session.rollback()

                        app.logger.exception(

                            "WhatsApp AI processing "
                            "failed for %s",

                            from_phone
                        )

                        continue

                    send_whatsapp_message(

                        from_phone,

                        reply_text
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