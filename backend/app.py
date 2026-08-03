import os

from flask import Flask, render_template, request, redirect, session

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from database.db import db


from models.user import User
from models.business import Business
from models.menu import Menu
from models.order import Order
from models.order_item import OrderItem
from models.customer import Customer
from models.conversation import Conversation
from models.pending_order import PendingOrder



from services.ai.prompt_builder import build_restaurant_prompt
from services.ai.openai_provider import OpenAIProvider
from services.ai.order_modifier import (
    interpret_order_request
)

from services.ai.order_executor import (
    execute_order_action
)
from services.ai.order_extractor import extract_order

from services.ai.agent import run_agent



app = Flask(__name__)


app.secret_key = "my_super_secret_key"



app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///business.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False



db.init_app(app)


# ==========================
# WHATSAPP CONFIG
# ==========================

WHATSAPP_VERIFY_TOKEN = os.environ.get(
    "WHATSAPP_VERIFY_TOKEN",
    "my_verify_token"
)

WHATSAPP_ACCESS_TOKEN = os.environ.get(
    "WHATSAPP_ACCESS_TOKEN",
    ""
)

WHATSAPP_PHONE_NUMBER_ID = os.environ.get(
    "WHATSAPP_PHONE_NUMBER_ID",
    ""
)


def send_whatsapp_message(to_phone, message_text):
    """
    Sends a text reply back to a customer via the WhatsApp
    Cloud API. Silently no-ops if credentials are not configured,
    so local development / testing never breaks because of this.
    """

    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        return

    try:

        import requests

        url = (
            f"https://graph.facebook.com/v20.0/"
            f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
        )

        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {
                "body": message_text
            }
        }

        requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10
        )

    except Exception:

        # Never let a delivery failure break the webhook response
        pass


# ==========================
# HOME
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================
# LOGIN
# ==========================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user_id"] = user.id

            return redirect(
                "/dashboard"
            )

        return "Invalid email or password."

    return render_template(
        "login.html"
    )


# ==========================
# REGISTER
# ==========================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = request.form["username"]

        email = request.form["email"]

        password = request.form["password"]

        language = request.form["language"]

        if User.query.filter_by(
            username=username
        ).first():

            return "Username already exists."

        if User.query.filter_by(
            email=email
        ).first():

            return "Email already exists."

        user = User(

            username=username,

            email=email,

            password=generate_password_hash(
                password
            ),

            language=language

        )

        db.session.add(user)

        db.session.commit()

        return redirect(
            "/login"
        )

    return render_template(
        "register.html"
    )


# ==========================
# DASHBOARD
# ==========================

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


# ==========================
# CREATE BUSINESS
# ==========================

@app.route(
    "/create-business",
    methods=["GET", "POST"]
)
def create_business():

    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        business = Business(

            name=request.form.get("name"),

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

        db.session.add(business)

        db.session.commit()

        return redirect(
            "/dashboard"
        )

    return render_template(
        "create_business.html"
    )


# ==========================
# ORDER MANAGEMENT HELPERS
# ==========================

def recalculate_order_total(order):
    total = 0

    for item in order.items:

        quantity = int(item.quantity or 0)
        price = float(item.price or 0)

        item.subtotal = price * quantity
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


# ==========================
# BUSINESS DASHBOARD
# ==========================

@app.route(
    "/business/<int:business_id>"
)
def business_dashboard(business_id):

    if "user_id" not in session:
        return redirect("/login")

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:
        return "Business not found", 404

    # ==========================
    # MENU
    # ==========================

    menu_items = Menu.query.filter_by(
        business_id=business.id
    ).all()

    # ==========================
    # ORDERS
    # ==========================

    orders = Order.query.filter_by(
        business_id=business.id
    ).order_by(
        Order.id.desc()
    ).all()

    recent_orders = orders[:8]

    # ==========================
    # CUSTOMERS
    # ==========================

    customers = Customer.query.filter_by(
        business_id=business.id
    ).all()

    # ==========================
    # CORE KPI ANALYTICS
    # ==========================

    valid_orders = [
        order
        for order in orders
        if order.status != "Cancelled"
    ]

    total_orders = len(orders)
    total_customers = len(customers)

    total_revenue = sum(
        float(order.total_price or 0)
        for order in valid_orders
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

    active_orders = (
        pending_orders
        + preparing_orders
        + ready_orders
    )

    average_order_value = (
        total_revenue / len(valid_orders)
        if valid_orders
        else 0
    )

    # ==========================
    # TOP SELLING ITEMS
    # ==========================

    item_sales = {}

    for order in valid_orders:

        for item in order.items:

            name = item.name

            if name not in item_sales:

                item_sales[name] = {
                    "name": name,
                    "quantity": 0,
                    "revenue": 0
                }

            item_sales[name]["quantity"] += int(
                item.quantity or 0
            )

            item_sales[name]["revenue"] += float(
                item.subtotal or 0
            )

    top_items = sorted(
        item_sales.values(),
        key=lambda item: item["quantity"],
        reverse=True
    )[:5]

    # ==========================
    # ORDER STATUS
    # ==========================

    order_status = {
        "Pending": pending_orders,
        "Preparing": preparing_orders,
        "Ready": ready_orders,
        "Completed": completed_orders,
        "Cancelled": cancelled_orders
    }

    # ==========================
    # 7-DAY REVENUE
    # ==========================

    from datetime import datetime, timedelta

    today = datetime.utcnow().date()
    revenue_chart = []

    for days_ago in range(6, -1, -1):

        chart_date = today - timedelta(
            days=days_ago
        )

        daily_revenue = 0

        for order in valid_orders:

            if not order.created_at:
                continue

            if order.created_at.date() == chart_date:
                daily_revenue += float(
                    order.total_price or 0
                )

        revenue_chart.append({
            "date": chart_date.strftime("%a"),
            "full_date": chart_date.strftime("%d %b"),
            "revenue": daily_revenue
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
        active_orders=active_orders,
        average_order_value=average_order_value,
        top_items=top_items,
        revenue_chart=revenue_chart,
        order_status=order_status,
        revenue=total_revenue
    )


# ==========================
# MENU INTELLIGENCE API
# ==========================

@app.route(
    "/business/<int:business_id>/menu/search"
)
def menu_search_api(business_id):

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


# ==========================
# MODIFY ORDER
# ==========================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/modify",
    methods=["POST"]
)
def modify_order(
    business_id,
    order_id
):

    if "user_id" not in session:
        return redirect("/login")

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

    if not order_can_be_modified(order):

        return (
            "This order can no longer be modified.",
            400
        )

    item_id = request.form.get("item_id")

    action = request.form.get(
        "action",
        "update"
    )

    try:

        item_id = int(item_id)

    except (TypeError, ValueError):

        return "Invalid order item.", 400

    item = OrderItem.query.filter_by(
        id=item_id,
        order_id=order.id
    ).first()

    if not item:
        return "Order item not found.", 404

    if action == "remove":

        db.session.delete(item)

        db.session.flush()

        remaining_items = OrderItem.query.filter_by(
            order_id=order.id
        ).all()

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

        return "Invalid modification action.", 400

    db.session.commit()

    return redirect(
        f"/business/{business.id}/orders"
    )


# ==========================
# CANCEL ORDER
# ==========================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/cancel",
    methods=["POST"]
)
def cancel_order(
    business_id,
    order_id
):

    if "user_id" not in session:
        return redirect("/login")

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

    if not order_can_be_cancelled(order):

        return (
            "This order cannot be cancelled.",
            400
        )

    order.status = "Cancelled"

    db.session.commit()

    return redirect(
        f"/business/{business.id}/orders"
    )


# ==========================
# VIEW ORDERS
# ==========================

@app.route(
    "/business/<int:business_id>/orders"
)
def view_orders(business_id):

    if "user_id" not in session:
        return redirect("/login")

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:
        return "Business not found", 404

    orders = Order.query.filter_by(
        business_id=business.id
    ).order_by(
        Order.id.desc()
    ).all()

    return render_template(
        "orders.html",
        business=business,
        orders=orders
    )


# ==========================
# UPDATE ORDER STATUS
# ==========================

@app.route(
    "/business/<int:business_id>/orders/<int:order_id>/status",
    methods=["POST"]
)
def update_order_status(business_id, order_id):

    if "user_id" not in session:
        return redirect("/login")

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

    new_status = request.form.get("status", "").strip()

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


# ==========================
# CREATE MANUAL ORDER
# ==========================

@app.route(

    "/business/<int:business_id>/orders/create",

    methods=["GET", "POST"]

)
def create_order(
    business_id
):

    if "user_id" not in session:

        return redirect("/login")

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

            business_id=business.id

        )

        db.session.add(order)

        db.session.commit()

        return redirect(

            f"/business/{business.id}/orders"

        )

    return render_template(

        "create_order.html",

        business=business

    )


# ==========================
# CREATE MENU ITEM
# ==========================

@app.route(
    "/business/<int:business_id>/menu/create",
    methods=["GET", "POST"]
)
def create_menu(business_id):

    if "user_id" not in session:
        return redirect("/login")

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:
        return "Business not found"

    if request.method == "POST":

        menu = Menu(

            name=request.form.get("name"),

            description=request.form.get(
                "description"
            ),

            price=float(
                request.form.get("price")
            ),

            category=request.form.get(
                "category"
            ),

            available=True,

            business_id=business.id

        )

        db.session.add(menu)

        db.session.commit()

        return redirect(
            f"/business/{business.id}"
        )

    return render_template(
        "create_menu.html",
        business=business
    )


# ==========================
# EDIT MENU
# ==========================

@app.route(
    "/menu/<int:menu_id>/edit",
    methods=["GET", "POST"]
)
def edit_menu(menu_id):

    if "user_id" not in session:
        return redirect("/login")

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


# ==========================
# DELETE MENU
# ==========================

@app.route(
    "/menu/<int:menu_id>/delete"
)
def delete_menu(menu_id):

    if "user_id" not in session:

        return redirect("/login")

    menu = Menu.query.get_or_404(
        menu_id
    )

    business = Business.query.filter_by(

        id=menu.business_id,

        owner_id=session["user_id"]

    ).first()

    if not business:

        return "Unauthorized"

    db.session.delete(menu)

    db.session.commit()

    return redirect(
        f"/business/{business.id}"
    )


# ==========================
# CUSTOMERS CRM
# ==========================

@app.route(
    "/business/<int:business_id>/customers",
    methods=["GET", "POST"]
)
def customers(business_id):

    if "user_id" not in session:
        return redirect("/login")

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
            return "Phone number is required.", 400

        existing_customer = Customer.query.filter_by(
            phone=phone,
            business_id=business.id
        ).first()

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

        db.session.add(customer)
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

    customers_list = query.order_by(
        Customer.id.desc()
    ).all()

    from sqlalchemy import func

    order_stats = db.session.query(
        Order.customer_id,
        func.count(Order.id).label(
            "order_count"
        ),
        func.coalesce(
            func.sum(Order.total_price),
            0
        ).label(
            "total_spent"
        ),
        func.max(Order.id).label(
            "last_order_id"
        )
    ).filter(
        Order.business_id == business.id
    ).group_by(
        Order.customer_id
    ).all()

    customer_stats = {
        row.customer_id: {
            "orders": row.order_count,
            "spent": float(
                row.total_spent or 0
            ),
            "last_order_id": row.last_order_id
        }
        for row in order_stats
    }

    total_customers = Customer.query.filter_by(
        business_id=business.id
    ).count()

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


# ==========================
# RESTAURANT SETTINGS
# ==========================

@app.route(
    "/business/<int:business_id>/settings",
    methods=["GET", "POST"]
)
def business_settings(business_id):

    if "user_id" not in session:
        return redirect("/login")

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
            return "Business name is required.", 400

        if not business_type:
            return "Business type is required.", 400

        business.name = name
        business.business_type = business_type
        business.address = address or None
        business.phone = phone or None

        db.session.commit()

        return redirect(
            f"/business/{business.id}/settings"
        )

    return render_template(
        "business_settings.html",
        business=business
    )


# ==========================
# CUSTOMER PROFILE
# ==========================

@app.route(
    "/business/<int:business_id>/customer/<int:customer_id>"
)
def customer_profile(
    business_id,
    customer_id
):

    if "user_id" not in session:
        return redirect("/login")

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

    # --------------------------
    # ORDER HISTORY
    # --------------------------

    orders = Order.query.filter_by(
        customer_id=customer.id,
        business_id=business.id
    ).order_by(
        Order.id.desc()
    ).limit(20).all()

    total_orders = Order.query.filter_by(
        customer_id=customer.id,
        business_id=business.id
    ).count()

    from sqlalchemy import func

    total_spent = db.session.query(
        func.coalesce(
            func.sum(Order.total_price),
            0
        )
    ).filter(
        Order.customer_id == customer.id,
        Order.business_id == business.id
    ).scalar()

    # --------------------------
    # CONVERSATION HISTORY
    # --------------------------

    conversations = Conversation.query.filter_by(
        customer_id=customer.id
    ).order_by(
        Conversation.id.desc()
    ).limit(20).all()

    total_conversations = Conversation.query.filter_by(
        customer_id=customer.id
    ).count()

    latest_order = orders[0] if orders else None

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


# ==========================
# AI PLAYGROUND
# ==========================

@app.route(
    "/business/<int:business_id>/ai",
    methods=["GET", "POST"]
)
def ai_playground(business_id):

    if "user_id" not in session:
        return redirect("/login")

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

    # ==========================
    # PROCESS CUSTOMER MESSAGE
    # ==========================

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

            # ==========================
            # FIND OR CREATE CUSTOMER
            # ==========================

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

                db.session.add(customer)
                db.session.commit()

            # ==========================
            # LOAD CONVERSATION MEMORY
            # ==========================

            previous_conversations = Conversation.query.filter_by(
                customer_id=customer.id
            ).order_by(
                Conversation.id.desc()
            ).limit(12).all()

            history = [
                {
                    "message": chat.message,
                    "response": chat.response
                }
                for chat in reversed(
                    previous_conversations
                )
            ]

            # ==========================
            # RUN CENTRAL AI AGENT
            # ==========================

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

                    frontend_action = agent_result.get(
                        "action"
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

                response = (
                    "AI agent error. "
                    "Please try again."
                )

            # ==========================
            # SAVE CONVERSATION
            # ==========================

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

    # ==========================
    # RELOAD CUSTOMER HISTORY
    # ==========================

    if customer:

        conversations = Conversation.query.filter_by(
            customer_id=customer.id
        ).order_by(
            Conversation.id.asc()
        ).all()

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


# ==========================
# AI ORDER CREATOR
# ==========================

@app.route(
    "/business/<int:business_id>/ai-order",
    methods=["GET", "POST"]
)
def ai_order(business_id):

    if "user_id" not in session:
        return redirect("/login")

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:
        return "Business not found."

    result = None

    phone = ""
    message = ""

    # ==========================
    # HANDLE FORM
    # ==========================

    if request.method == "POST":

        action = request.form.get("action", "generate")

        # ==========================
        # GENERATE ORDER
        # ==========================

        if action == "generate":

            phone = request.form.get("phone", "").strip()
            message = request.form.get("message", "").strip()

            if not phone or not message:

                result = {
                    "error": "Please enter the customer phone and message."
                }

            else:

                try:

                    extracted = extract_order(
                        business.id,
                        message
                    )

                    # No valid menu items found
                    if not extracted.get("items"):

                        result = {
                            "error": (
                                "I couldn't find any available menu items "
                                "in that request."
                            )
                        }

                    else:

                        result = {
                            "order": extracted,
                            "total": extracted["total"],
                            "restaurant": business.name,
                            "phone": phone,
                            "message": message,
                            "confirmed": False
                        }

                        # Save temporary order in session
                        session["pending_ai_order"] = {
                            "business_id": business.id,
                            "phone": phone,
                            "message": message,
                            "items": extracted["items"],
                            "total": extracted["total"]
                        }

                        session.modified = True

                except Exception as e:

                    result = {
                        "error": f"Unable to generate order: {str(e)}"
                    }

        # ==========================
        # CONFIRM ORDER
        # ==========================

        elif action == "confirm":

            pending = session.get("pending_ai_order")

            if not pending:

                result = {
                    "error": "No pending order to confirm."
                }

            elif pending.get("business_id") != business.id:

                result = {
                    "error": "This order does not belong to this business."
                }

            else:

                try:

                    phone = pending["phone"]

                    # Find existing customer
                    customer = Customer.query.filter_by(
                        phone=phone,
                        business_id=business.id
                    ).first()

                    # Create customer if needed
                    if not customer:

                        customer = Customer(
                            name="New Customer",
                            phone=phone,
                            language="English",
                            business_id=business.id
                        )

                        db.session.add(customer)
                        db.session.flush()

                    # ==========================
                    # RE-CALCULATE FROM REAL MENU
                    # ==========================

                    final_items = []
                    final_total = 0

                    for item in pending["items"]:

                        menu = Menu.query.filter(
                            Menu.business_id == business.id,
                            Menu.name == item["name"],
                            Menu.available == True
                        ).first()

                        if not menu:
                            continue

                        quantity = int(item["quantity"])

                        subtotal = (
                            float(menu.price) *
                            quantity
                        )

                        final_items.append({
                            "name": menu.name,
                            "quantity": quantity,
                            "price": float(menu.price),
                            "subtotal": subtotal
                        })

                        final_total += subtotal

                    if not final_items:

                        db.session.rollback()

                        result = {
                            "error": (
                                "The menu items in this order are no "
                                "longer available."
                            )
                        }

                    else:

                        # ==========================
                        # CREATE ORDER
                        # ==========================

                        order = Order(
                            customer_name=customer.name,
                            customer_phone=customer.phone,
                            delivery_address="Unknown",
                            total_price=final_total,
                            status="Pending",
                            business_id=business.id,
                            customer_id=customer.id
                        )

                        db.session.add(order)
                        db.session.flush()

                        # ==========================
                        # CREATE ORDER ITEMS
                        # ==========================

                        for item in final_items:

                            order_item = OrderItem(
                                name=item["name"],
                                quantity=item["quantity"],
                                price=item["price"],
                                subtotal=item["subtotal"],
                                order_id=order.id
                            )

                            db.session.add(order_item)

                        db.session.commit()

                        # Clear pending order
                        session.pop(
                            "pending_ai_order",
                            None
                        )

                        result = {
                            "confirmed": True,
                            "order_id": order.id,
                            "restaurant": business.name,
                            "phone": customer.phone,
                            "items": final_items,
                            "total": final_total
                        }

                except Exception as e:

                    db.session.rollback()

                    result = {
                        "error": (
                            f"Unable to confirm order: {str(e)}"
                        )
                    }

    # ==========================
    # LOAD PENDING ORDER
    # ==========================

    pending = session.get("pending_ai_order")

    if pending and not result:

        result = {
            "order": {
                "items": pending["items"],
                "total": pending["total"]
            },
            "total": pending["total"],
            "restaurant": business.name,
            "phone": pending["phone"],
            "message": pending["message"],
            "confirmed": False
        }

        phone = pending["phone"]
        message = pending["message"]

    return render_template(
        "ai_order.html",
        business=business,
        result=result,
        phone=phone,
        message=message
    )


# ==========================
# AI ORDER MODIFICATION
# ==========================

@app.route(
    "/business/<int:business_id>/ai-order/modify",
    methods=["POST"]
)
def ai_order_modify(
    business_id
):

    if "user_id" not in session:
        return redirect("/login")

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

    # --------------------------
    # FIND CUSTOMER
    # --------------------------

    customer = Customer.query.filter_by(
        phone=phone,
        business_id=business.id
    ).first()

    if not customer:

        return {
            "success": False,
            "message": (
                "Customer not found."
            )
        }, 404

    # --------------------------
    # FIND ACTIVE ORDER
    # --------------------------

    order = Order.query.filter(
        Order.customer_id == customer.id,
        Order.business_id == business.id,
        Order.status.in_([
            "Pending",
            "Preparing"
        ])
    ).order_by(
        Order.id.desc()
    ).first()

    if not order:

        return {
            "success": False,
            "message": (
                "No active order was found "
                "for this customer."
            )
        }, 404

    # --------------------------
    # AI INTERPRETATION
    # --------------------------

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

    # --------------------------
    # UNKNOWN REQUEST
    # --------------------------

    if command.get("action") == "unknown":

        return {
            "success": False,
            "message": (
                "I couldn't understand the "
                "requested order change."
            ),
            "command": command
        }, 400

    # --------------------------
    # EXECUTE
    # --------------------------

    try:

        result = execute_order_action(
            business.id,
            order,
            command
        )

        result["order_id"] = order.id

        return result

    except Exception as e:

        db.session.rollback()

        return {
            "success": False,
            "message": (
                f"Unable to modify order: {str(e)}"
            )
        }, 500


# ==========================
# UNIFIED AI AGENT API
# ==========================

@app.route(
    "/api/agent/chat",
    methods=["POST"]
)
def agent_chat():

    if "user_id" not in session:
        return {
            "success": False,
            "error": "Unauthorized"
        }, 401

    data = request.get_json(
        silent=True
    )

    if not isinstance(data, dict):

        return {
            "success": False,
            "error": "Invalid JSON request."
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
            "error": "business_id is required."
        }, 400

    if not phone:

        return {
            "success": False,
            "error": "phone is required."
        }, 400

    if not message:

        return {
            "success": False,
            "error": "message is required."
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
            "error": "Invalid business_id."
        }, 400

    # ==========================
    # BUSINESS ACCESS
    # ==========================

    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()

    if not business:

        return {
            "success": False,
            "error": "Business not found."
        }, 404

    # ==========================
    # CUSTOMER
    # ==========================

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

    # ==========================
    # LANGUAGE
    # ==========================

    language = (
        customer.language
        or "English"
    )

    # ==========================
    # LOAD CONVERSATION MEMORY
    # ==========================

    previous_conversations = Conversation.query.filter_by(
        customer_id=customer.id
    ).order_by(
        Conversation.id.desc()
    ).limit(12).all()

    history = [
        {
            "message": chat.message,
            "response": chat.response
        }
        for chat in reversed(
            previous_conversations
        )
    ]

    # ==========================
    # RUN AGENT
    # ==========================

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
            "error": "Invalid agent response."
        }, 500

    # ==========================
    # SAVE CONVERSATION
    # ==========================

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

    # ==========================
    # RESPONSE
    # ==========================

    return {
        "success": True,
        "type": result.get(
            "type",
            "response"
        ),
        "message": response_message,
        "action": result.get(
            "action"
        ),
        "business_id": business.id,
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "language": customer.language
        }
    }


# ==========================
# WHATSAPP WEBHOOK
# ==========================

@app.route(
    "/webhook/whatsapp",
    methods=["GET", "POST"]
)
def whatsapp_webhook():

    # ==========================
    # META WEBHOOK VERIFICATION
    # ==========================

    if request.method == "GET":

        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            return challenge, 200

        return "Verification failed", 403

    # ==========================
    # INCOMING MESSAGE(S)
    # ==========================

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return "OK", 200

    try:

        entries = data.get("entry", [])

        for entry in entries:

            changes = entry.get("changes", [])

            for change in changes:

                value = change.get("value", {})

                messages = value.get("messages", [])

                if not messages:
                    continue

                contacts = value.get("contacts", [])

                # NOTE: this maps every incoming message to your first
                # registered business. If you run more than one
                # restaurant on WhatsApp, add a column (e.g.
                # whatsapp_phone_number_id) to the Business model and
                # look the business up from value["metadata"] instead.
                business = Business.query.order_by(
                    Business.id.asc()
                ).first()

                if not business:
                    continue

                for wa_message in messages:

                    if wa_message.get("type") != "text":
                        continue

                    from_phone = wa_message.get("from")

                    text_body = wa_message.get(
                        "text", {}
                    ).get("body", "").strip()

                    if not from_phone or not text_body:
                        continue

                    contact_name = "New Customer"

                    for contact in contacts:

                        if contact.get("wa_id") == from_phone:

                            contact_name = contact.get(
                                "profile", {}
                            ).get(
                                "name",
                                "New Customer"
                            )

                    # ==========================
                    # FIND OR CREATE CUSTOMER
                    # ==========================

                    customer = Customer.query.filter_by(
                        phone=from_phone,
                        business_id=business.id
                    ).first()

                    if not customer:

                        customer = Customer(
                            name=contact_name,
                            phone=from_phone,
                            language="English",
                            business_id=business.id
                        )

                        db.session.add(customer)
                        db.session.commit()

                    # ==========================
                    # LOAD CONVERSATION MEMORY
                    # ==========================

                    previous_conversations = Conversation.query.filter_by(
                        customer_id=customer.id
                    ).order_by(
                        Conversation.id.desc()
                    ).limit(12).all()

                    history = [
                        {
                            "message": chat.message,
                            "response": chat.response
                        }
                        for chat in reversed(
                            previous_conversations
                        )
                    ]

                    # ==========================
                    # RUN AI AGENT
                    # ==========================

                    try:

                        agent_result = run_agent(
                            business.id,
                            customer.phone,
                            text_body,
                            customer.language,
                            history
                        )

                    except Exception:

                        db.session.rollback()
                        agent_result = None

                    if isinstance(agent_result, dict):

                        reply_text = agent_result.get("message") or (
                            "Sorry, I couldn't process that right now."
                        )

                    else:

                        reply_text = (
                            "Sorry, I couldn't process that right now."
                        )

                    # ==========================
                    # SAVE CONVERSATION
                    # ==========================

                    conversation = Conversation(
                        customer_id=customer.id,
                        message=text_body,
                        response=reply_text
                    )

                    db.session.add(conversation)

                    try:

                        db.session.commit()

                    except Exception:

                        db.session.rollback()

                    # ==========================
                    # SEND REPLY
                    # ==========================

                    send_whatsapp_message(
                        from_phone,
                        reply_text
                    )

    except Exception:

        db.session.rollback()

    return "OK", 200


# ==========================
# LOGOUT
# ==========================

@app.route("/logout")
def logout():

    session.pop(
        "user_id",
        None
    )

    return redirect("/login")


# ==========================
# ABOUT
# ==========================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ==========================
# ERROR HANDLERS
# ==========================

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


# ==========================
# CONTEXT PROCESSOR
# ==========================
# Makes global data available
# in every template

@app.context_processor
def inject_globals():

    return {

        "app_name":
        "Botify AI",

        "currency":
        "FCFA"

    }


# ==========================
# START APPLICATION
# ==========================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )
