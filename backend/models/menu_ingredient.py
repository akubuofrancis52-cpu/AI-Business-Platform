from database.db import db


class MenuIngredient(db.Model):
    __tablename__ = "menu_ingredients"

    id = db.Column(
        db.Integer,
        primary_key=True,
    )

    menu_id = db.Column(
        db.Integer,
        db.ForeignKey("menu.id"),
        nullable=False,
        index=True,
    )

    ingredient_id = db.Column(
        db.Integer,
        db.ForeignKey("ingredients.id"),
        nullable=False,
        index=True,
    )

    quantity_required = db.Column(
        db.Float,
        nullable=False,
        default=1,
    )

    menu = db.relationship(
        "Menu",
        backref=db.backref(
            "ingredients",
            lazy=True,
            cascade="all, delete-orphan",
        ),
    )

    ingredient = db.relationship(
        "Ingredient",
        backref=db.backref(
            "menu_items",
            lazy=True,
        ),
    )

    def __repr__(self):
        return (
            f"<MenuIngredient "
            f"menu={self.menu_id} "
            f"ingredient={self.ingredient_id}>"
        )
