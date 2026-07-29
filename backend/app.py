from flask import Flask, render_template, request, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

from database.db import db

from models.user import User
from models.business import Business
from models.menu import Menu
from models.order import Order
from models.customer import Customer
from models.conversation import Conversation

from services.ai.prompt_builder import build_restaurant_prompt
from services.ai.openai_provider import OpenAIProvider
from services.ai.order_extractor import extract_order


app = Flask(__name__)

app.secret_key = "my_super_secret_key"


app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///business.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db.init_app(app)



# ==========================
# HOME
# ==========================

@app.route("/")
def home():

    return render_template("index.html")



# ==========================
# LOGIN
# ==========================

@app.route("/login", methods=["GET", "POST"])
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

            return redirect("/dashboard")


        return "Invalid email or password."


    return render_template("login.html")



# ==========================
# REGISTER
# ==========================

@app.route("/register", methods=["GET", "POST"])
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
            password=generate_password_hash(password),
            language=language
        )


        db.session.add(user)
        db.session.commit()


        return redirect("/login")


    return render_template("register.html")



# ==========================
# DASHBOARD
# ==========================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")


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

@app.route("/create-business", methods=["GET", "POST"])
def create_business():

    if "user_id" not in session:
        return redirect("/login")


    if request.method == "POST":

        business = Business(
            name=request.form["name"],
            business_type=request.form["business_type"],
            address=request.form["address"],
            phone=request.form["phone"],
            owner_id=session["user_id"]
        )


        db.session.add(business)

        db.session.commit()


        return redirect("/dashboard")


    return render_template(
        "create_business.html"
    )



# ==========================
# BUSINESS DASHBOARD
# ==========================

@app.route("/business/<int:business_id>")
def business_dashboard(business_id):

    if "user_id" not in session:
        return redirect("/login")


    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()


    if not business:
        return "Business not found."


    menu_items = Menu.query.filter_by(
        business_id=business.id
    ).all()


    return render_template(
        "business_dashboard.html",
        business=business,
        menu_items=menu_items
    )



# ==========================
# ORDERS
# ==========================

@app.route("/business/<int:business_id>/orders")
def view_orders(business_id):

    if "user_id" not in session:
        return redirect("/login")


    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()


    if not business:
        return "Business not found."


    orders = Order.query.filter_by(
        business_id=business.id
    ).all()


    return render_template(
        "orders.html",
        business=business,
        orders=orders
    )



@app.route(
    "/business/<int:business_id>/orders/create",
    methods=["GET", "POST"]
)
def create_order(business_id):

    if "user_id" not in session:
        return redirect("/login")


    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()


    if not business:
        return "Business not found."


    if request.method == "POST":

        order = Order(
            customer_name=request.form["customer_name"],
            customer_phone=request.form["customer_phone"],
            delivery_address=request.form["delivery_address"],
            total_price=float(
                request.form["total_price"]
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
        return "Business not found."


    if request.method == "POST":

        menu = Menu(
            name=request.form["name"],
            description=request.form["description"],
            price=float(request.form["price"]),
            category=request.form["category"],
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


    menu = Menu.query.get_or_404(menu_id)


    business = Business.query.filter_by(
        id=menu.business_id,
        owner_id=session["user_id"]
    ).first()


    if not business:
        return "Unauthorized."


    if request.method == "POST":

        menu.name = request.form["name"]
        menu.description = request.form["description"]
        menu.price = float(request.form["price"])
        menu.category = request.form["category"]


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

@app.route("/menu/<int:menu_id>/delete")
def delete_menu(menu_id):

    if "user_id" not in session:
        return redirect("/login")


    menu = Menu.query.get_or_404(menu_id)


    business = Business.query.filter_by(
        id=menu.business_id,
        owner_id=session["user_id"]
    ).first()


    if not business:
        return "Unauthorized."


    db.session.delete(menu)

    db.session.commit()


    return redirect(
        f"/business/{business.id}"
    )



# ==========================
# CUSTOMERS
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
        return "Business not found."


    if request.method == "POST":

        customer = Customer(
            name=request.form["name"],
            phone=request.form["phone"],
            language=request.form["language"],
            business_id=business.id
        )


        db.session.add(customer)

        db.session.commit()


        return redirect(
            f"/business/{business.id}/customers"
        )


    customers = Customer.query.filter_by(
        business_id=business.id
    ).all()


    return render_template(
        "customers.html",
        business=business,
        customers=customers
    )



# ==========================
# CUSTOMER PROFILE
# ==========================

@app.route(
    "/business/<int:business_id>/customer/<int:customer_id>"
)
def customer_profile(business_id, customer_id):

    if "user_id" not in session:
        return redirect("/login")


    business = Business.query.filter_by(
        id=business_id,
        owner_id=session["user_id"]
    ).first()


    customer = Customer.query.filter_by(
        id=customer_id,
        business_id=business.id
    ).first()


    if not customer:
        return "Customer not found."


    return render_template(
        "customer_profile.html",
        business=business,
        customer=customer
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
        return "Business not found."


    response = None


    if request.method == "POST":

        message = request.form.get("message")
        phone = request.form.get("phone")


        if not message or not phone:

            response = "Please enter customer phone and message."

        else:


            customer = Customer.query.filter_by(
                phone=phone,
                business_id=business.id
            ).first()



            if not customer:

                customer = Customer(
                    name="New Customer",
                    phone=phone,
                    language="English",
                    business_id=business.id
                )

                db.session.add(customer)
                db.session.commit()



            conversations = Conversation.query.filter_by(
                customer_id=customer.id
            ).order_by(
                Conversation.id.desc()
            ).limit(10).all()



            history = ""


            for chat in reversed(conversations):

                history += f"""
Customer:
{chat.message}

AI:
{chat.response}

"""


            prompt = build_restaurant_prompt(
                business.id,
                message
            )


            prompt += f"""

Conversation history:

{history}


Answer like a professional restaurant AI assistant.
Help customers with menu, prices, orders and support.
"""


            try:

                ai = OpenAIProvider()

                response = ai.generate(prompt)


            except Exception as e:

                response = str(e)



            conversation = Conversation(
                customer_id=customer.id,
                message=message,
                response=response
            )


            db.session.add(conversation)

            db.session.commit()



    return render_template(
        "ai_playground.html",
        business=business,
        response=response
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


    if request.method == "POST":

        phone = request.form["phone"]
        message = request.form["message"]


        customer = Customer.query.filter_by(
            phone=phone,
            business_id=business.id
        ).first()


        if not customer:

            customer = Customer(
                name="New Customer",
                phone=phone,
                language="English",
                business_id=business.id
            )

            db.session.add(customer)
            db.session.commit()



        extracted = extract_order(
            business.id,
            message
        )


        total = 0


        for item in extracted["items"]:

            menu = Menu.query.filter_by(
                business_id=business.id,
                name=item["name"]
            ).first()


            if menu:

                total += menu.price * item["quantity"]



        order = Order(
            customer_name=customer.name,
            customer_phone=customer.phone,
            delivery_address="Unknown",
            total_price=total,
            status="Pending",
            business_id=business.id,
            customer_id=customer.id
        )


        db.session.add(order)

        db.session.commit()


        result = {
            "order": extracted,
            "total": total
        }



    return render_template(
        "ai_order.html",
        business=business,
        result=result
    )



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
# START APP
# ==========================

if __name__ == "__main__":

    with app.app_context():

        db.create_all()


    app.run(
        debug=True
    )