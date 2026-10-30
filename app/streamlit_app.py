from pathlib import Path
import json
import pandas as pd
import streamlit as st


# -------------------------------------------------
# Paths
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

PRODUCTS_PATH = DATA_DIR / "products.csv"
RULES_PATH = MODEL_DIR / "rules.json"
METADATA_PATH = MODEL_DIR / "metadata.json"


# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Boréal Marché Recommendation Engine",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Boréal Marché")
st.caption("Association-rule recommendation engine using Apriori")


# -------------------------------------------------
# Load data
# -------------------------------------------------

@st.cache_data
def load_products():
    products = pd.read_csv(PRODUCTS_PATH)

    required = {
        "Item",
        "Item_EN",
        "Category",
        "Price"
    }

    missing = required - set(products.columns)

    if missing:
        raise ValueError(
            f"Missing product columns: {missing}"
        )

    return products


@st.cache_data
def load_rules():
    with open(
        RULES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@st.cache_data
def load_metadata():
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


products = load_products()
rules = load_rules()
metadata = load_metadata()


# -------------------------------------------------
# Session state
# -------------------------------------------------

if "basket" not in st.session_state:
    st.session_state.basket = []


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def strength_label(lift):
    if lift >= 5:
        return "Very strong"
    elif lift >= 2:
        return "Strong"
    elif lift >= 1.25:
        return "Moderate"
    else:
        return "Weak"


def get_product(item):
    row = products[
        products["Item"] == item
    ]

    if row.empty:
        return None

    return row.iloc[0]


def popular_products(limit=5):
    """
    Cold-start fallback.

    Consequent support from the mined rules is used
    as an approximation of item popularity.
    """

    popularity = {}

    for rule in rules:
        consequent_support = rule.get(
            "consequent_support",
            0
        )

        for item in rule["consequents"]:
            popularity[item] = max(
                popularity.get(item, 0),
                consequent_support
            )

    ranked = sorted(
        popularity.items(),
        key=lambda x: x[1],
        reverse=True
    )

    output = []

    for item, score in ranked:
        if item in st.session_state.basket:
            continue

        product = get_product(item)

        if product is None:
            continue

        output.append(
            {
                "item": item,
                "score": score,
                "product": product
            }
        )

        if len(output) >= limit:
            break

    return output


def find_recommendations(
    basket_items,
    max_results=5
):
    basket_set = set(basket_items)

    best_by_product = {}
    fired_rules = []

    for rule in rules:

        antecedents = set(
            rule["antecedents"]
        )

        consequents = set(
            rule["consequents"]
        )

        # Rule fires if antecedent is a subset
        # of the current basket
        if not antecedents.issubset(
            basket_set
        ):
            continue

        fired_rules.append(rule)

        for item in consequents:

            # Never recommend an item
            # already present in the basket
            if item in basket_set:
                continue

            current = best_by_product.get(
                item
            )

            # If several rules recommend the same product,
            # keep the rule with the highest lift
            if (
                current is None
                or rule["lift"] > current["lift"]
            ):
                best_by_product[item] = rule

    ranked = sorted(
        best_by_product.items(),
        key=lambda x: x[1]["lift"],
        reverse=True
    )

    return (
        ranked[:max_results],
        fired_rules
    )


def explanation_text(
    rule,
    recommended_item
):
    antecedent_text = ", ".join(
        rule["antecedents"]
    )

    confidence_pct = (
        rule["confidence"] * 100
    )

    return (
        f"Because your basket contains "
        f"{antecedent_text}, customers who buy "
        f"these items also buy {recommended_item} "
        f"{confidence_pct:.1f}% of the time. "
        f"This is {rule['lift']:.2f}× more often "
        f"than average."
    )


# -------------------------------------------------
# Main layout
# -------------------------------------------------

catalogue_col, basket_col, reco_col = st.columns(
    [1.15, 0.9, 1.35],
    gap="large"
)


# =================================================
# Zone 1 — Catalogue
# =================================================

with catalogue_col:

    st.header("1 · Catalogue")

    search = st.text_input(
        "Search products"
    )

    categories = sorted(
        products["Category"]
        .dropna()
        .unique()
    )

    category = st.selectbox(
        "Category",
        ["All"] + categories
    )

    filtered = products.copy()

    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    if search:
        mask = (
            filtered["Item"]
            .str.contains(
                search,
                case=False,
                na=False
            )
            |
            filtered["Item_EN"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        )

        filtered = filtered[mask]

    st.caption(
        f"{len(filtered)} product(s)"
    )

    for _, row in filtered.iterrows():

        with st.container(
            border=True
        ):

            st.markdown(
                f"**{row['Item']}**"
            )

            st.caption(
                f"{row['Item_EN']} · "
                f"{row['Category']}"
            )

            st.write(
                f"${row['Price']:.2f}"
            )

            already_added = (
                row["Item"]
                in st.session_state.basket
            )

            if st.button(
                "Added ✓"
                if already_added
                else "Add",
                key=f"add_{row['Item']}",
                disabled=already_added,
                use_container_width=True
            ):

                st.session_state.basket.append(
                    row["Item"]
                )

                st.rerun()


# =================================================
# Zone 2 — Basket
# =================================================

with basket_col:

    st.header("2 · Basket")

    if not st.session_state.basket:

        st.info(
            "Your basket is empty."
        )

    else:

        total = 0.0

        for item in list(
            st.session_state.basket
        ):

            product = get_product(
                item
            )

            if product is None:
                continue

            total += float(
                product["Price"]
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{product['Item']}**"
                )

                st.caption(
                    f"{product['Item_EN']} · "
                    f"{product['Category']}"
                )

                st.write(
                    f"${product['Price']:.2f}"
                )

                if st.button(
                    "Remove",
                    key=f"remove_{item}",
                    use_container_width=True
                ):

                    st.session_state.basket.remove(
                        item
                    )

                    st.rerun()

        st.divider()

        st.metric(
            "Basket total",
            f"${total:.2f}"
        )

        if st.button(
            "Clear basket",
            type="secondary",
            use_container_width=True
        ):

            st.session_state.basket = []

            st.rerun()


# =================================================
# Zone 3 — Recommendations
# =================================================

with reco_col:

    st.header(
        "3 · Recommendations"
    )

    basket_items = (
        st.session_state.basket
    )

    if not basket_items:

        st.info(
            "Your basket is empty, so no association "
            "rule can fire. Showing popular products instead."
        )

        fallback = popular_products(
            limit=5
        )

        for result in fallback:

            product = result["product"]

            with st.container(
                border=True
            ):

                st.subheader(
                    product["Item"]
                )

                st.caption(
                    f"{product['Item_EN']} · "
                    f"{product['Category']}"
                )

                st.write(
                    f"${product['Price']:.2f}"
                )

                st.caption(
                    "Popular-product fallback "
                    "(cold start)"
                )

    else:

        recommendations, fired_rules = (
            find_recommendations(
                basket_items
            )
        )

        if not recommendations:

            st.warning(
                "No association rule matches "
                "the current basket. "
                "Showing popular products instead."
            )

            fallback = popular_products(
                limit=5
            )

            for result in fallback:

                product = (
                    result["product"]
                )

                with st.container(
                    border=True
                ):

                    st.subheader(
                        product["Item"]
                    )

                    st.caption(
                        f"{product['Item_EN']} · "
                        f"{product['Category']}"
                    )

                    st.write(
                        f"${product['Price']:.2f}"
                    )

                    st.caption(
                        "Fallback recommendation"
                    )

        else:

            for item, rule in recommendations:

                product = get_product(
                    item
                )

                if product is None:
                    continue

                strength = (
                    strength_label(
                        rule["lift"]
                    )
                )

                with st.container(
                    border=True
                ):

                    st.subheader(
                        product["Item"]
                    )

                    st.caption(
                        f"{product['Item_EN']} · "
                        f"{product['Category']} · "
                        f"${product['Price']:.2f}"
                    )

                    st.markdown(
                        f"**Strength:** {strength}"
                    )

                    st.write(
                        explanation_text(
                            rule,
                            item
                        )
                    )

                    metric_col1, metric_col2, metric_col3 = (
                        st.columns(3)
                    )

                    metric_col1.metric(
                        "Support",
                        f"{rule['support']:.3f}"
                    )

                    metric_col2.metric(
                        "Confidence",
                        f"{rule['confidence']:.1%}"
                    )

                    metric_col3.metric(
                        "Lift",
                        f"{rule['lift']:.2f}"
                    )

        # -----------------------------------------
        # Auditability
        # -----------------------------------------

        with st.expander(
            "Rules that fired"
        ):

            if not fired_rules:

                st.write(
                    "No rules fired."
                )

            else:

                fired_rows = []

                for rule in fired_rules:

                    fired_rows.append(
                        {
                            "Antecedent":
                                ", ".join(
                                    rule[
                                        "antecedents"
                                    ]
                                ),

                            "Consequent":
                                ", ".join(
                                    rule[
                                        "consequents"
                                    ]
                                ),

                            "Support":
                                rule["support"],

                            "Confidence":
                                rule[
                                    "confidence"
                                ],

                            "Lift":
                                rule["lift"]
                        }
                    )

                fired_df = pd.DataFrame(
                    fired_rows
                )

                fired_df = (
                    fired_df
                    .sort_values(
                        "Lift",
                        ascending=False
                    )
                )

                st.dataframe(
                    fired_df,
                    use_container_width=True,
                    hide_index=True
                )


# -------------------------------------------------
# Model information
# -------------------------------------------------

st.divider()

with st.expander(
    "Model information"
):

    st.json(metadata)