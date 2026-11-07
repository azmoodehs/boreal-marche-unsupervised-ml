from pathlib import Path
import json
from html import escape
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

st.markdown(
    """
    <style>
      .stApp {background: #f8faf7;}
      .block-container {max-width: 1480px; padding-top: 1.2rem; padding-bottom: 3rem;}
      h1,h2,h3 {letter-spacing: -.025em !important;}
      h1 {font-size: 2.35rem !important; color: #153f35 !important;}
      h3 {font-size: 1.05rem !important; padding: 0 !important;}
      [data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 16px !important;
          border-color: #e3eae3 !important;
          background: #fff;
      }
      div.stButton > button {border-radius: 9px; font-weight: 650; min-height: 2.4rem;}
      div.stButton > button[kind="primary"] {background: #176d55; border-color: #176d55; color: white;}
      div.stButton > button:hover {border-color: #176d55; color: #176d55;}
      div.stButton > button[kind="primary"]:hover {background: #10523f; color:white;}
      .shop-nav {display:flex; justify-content:space-between; align-items:center;
          padding:.55rem 0 1rem; border-bottom:1px solid #dce7df; margin-bottom:1.1rem;}
      .shop-brand {font-size:1.1rem; font-weight:850; color:#174a3c; letter-spacing:-.025em;}
      .shop-tag {background:#e5f3eb; color:#176d55; padding:.3rem .7rem; border-radius:999px;
          font-size:.74rem; font-weight:700;}
      .hero {background:linear-gradient(108deg,#164f3f 0%,#24795d 65%,#427f59 100%);
          color:white; border-radius:20px; padding:1.5rem 1.9rem 1.65rem; margin-bottom:1.4rem;}
      .hero-eyebrow {font-size:.72rem; letter-spacing:.15em; font-weight:750; color:#cfe8c7;}
      .hero-title {font-size:2rem; line-height:1.16; font-weight:800; margin:.55rem 0 .55rem;}
      .hero-copy {font-size:.91rem; color:#e1f0e8; max-width:660px;}
      .section-title {font-size:1.32rem; font-weight:790; color:#183d31; margin:.1rem 0 .4rem;}
      .muted-note {color:#728378; font-size:.84rem; margin-bottom:.65rem;}
      .product-emoji {background:#edf6ed; border-radius:12px; width:100%; text-align:center;
          font-size:2.15rem; padding:.48rem 0; margin-bottom:.5rem;}
      .product-name {font-size:.92rem; color:#233e34; font-weight:750; line-height:1.2; min-height:1.15rem;}
      .product-category {font-size:.75rem; color:#7b8b80; margin:.25rem 0;}
      .product-price {font-size:1rem; color:#176d55; font-weight:800; margin:.3rem 0 .45rem;}
      .cart-name {font-size:.88rem; font-weight:700; color:#244237;}
      .cart-note {font-size:.75rem; color:#77877e;}
      .reco-meta {color:#7b8b80; font-size:.77rem; margin:.1rem 0 .4rem;}
      .reco-stat {font-size:.76rem; color:#596e62; margin:.4rem 0;}
      .reco-stat b {color:#197054;}
      .promo {font-size:.79rem; color:#8c641d; background:#fff6e4;
          padding:.45rem .7rem; border-radius:9px; margin:.45rem 0;}
      [data-testid="stMetricValue"] {color:#165c46; font-size:1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="shop-nav"><span class="shop-brand">✳ Boréal Marché</span>'
    '<span class="shop-tag">✦ Smart grocery recommendations</span></div>'
    '<div class="hero"><div class="hero-eyebrow">DISCOVER MORE WITH EVERY BASKET</div>'
    '<div class="hero-title">Good food. Better discoveries.</div>'
    '<div class="hero-copy">Build your basket, find products that go well together, '
    'and see why each recommendation is a great match.</div></div>',
    unsafe_allow_html=True,
)

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


def product_icon(category):
    name = str(category).lower()
    if any(word in name for word in ("fruit", "veget", "produce", "légume")):
        return "🥬"
    if any(word in name for word in ("meat", "viande", "protein", "boucher")):
        return "🥩"
    if any(word in name for word in ("dairy", "lait", "cheese", "fromage")):
        return "🧀"
    if any(word in name for word in ("bread", "baker", "pain", "boulanger")):
        return "🥖"
    if any(word in name for word in ("drink", "beverage", "boisson")):
        return "🧃"
    if any(word in name for word in ("snack", "sweet", "dessert")):
        return "🍪"
    return "🛍️"


# -------------------------------------------------
# Main layout
# -------------------------------------------------

catalogue_col, basket_col, reco_col = st.columns(
    [1.1, 1.0, 1.1],
    gap="medium"
)


# =================================================
# Zone 1 — Catalogue
# =================================================

with catalogue_col:
    with st.container(border=True):

        st.subheader("01  ·  Catalogue")

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
    with st.container(border=True):

        st.subheader("02  ·  Basket")

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
    with st.container(border=True):

        st.subheader("03  ·  Recommendations")

        basket_items = (
            st.session_state.basket
        )

        if not basket_items:
            st.caption("Start with a suggestion or search the catalogue.")
            fallback = popular_products(limit=5)
            for result in fallback:
                item = result["item"]
                product = result["product"]
                with st.container(border=True):
                    name_col, action_col = st.columns([2.3, 1], gap="small", vertical_alignment="center")
                    with name_col:
                        st.markdown(f"**{item}**")
                        st.caption(f"{product['Category']} · ${product['Price']:.2f} · Popular pick")
                    with action_col:
                        if st.button("＋ Add", key=f"reco_cold_{item}", use_container_width=True):
                            if item not in st.session_state.basket:
                                st.session_state.basket.append(item)
                            st.rerun()
        else:
            recommendations, fired_rules = find_recommendations(basket_items)
            if not recommendations:
                st.caption("No matching association rule · Showing popular picks")
                for result in popular_products(limit=5):
                    item = result["item"]
                    product = result["product"]
                    with st.container(border=True):
                        name_col, action_col = st.columns([2.3, 1], gap="small", vertical_alignment="center")
                        with name_col:
                            st.markdown(f"**{item}**")
                            st.caption(f"{product['Category']} · ${product['Price']:.2f} · Popular pick")
                        with action_col:
                            if st.button("＋ Add", key=f"reco_fallback_{item}", use_container_width=True):
                                if item not in st.session_state.basket:
                                    st.session_state.basket.append(item)
                                st.rerun()
            else:
                st.caption("Ranked by association strength (Lift)")
                for item, rule in recommendations:
                    product = get_product(item)
                    if product is None:
                        continue
                    with st.container(border=True):
                        name_col, action_col = st.columns([2.3, 1], gap="small", vertical_alignment="center")
                        with name_col:
                            st.markdown(f"**{item}**")
                            st.caption(f"{product['Category']} · ${product['Price']:.2f} · {strength_label(rule['lift'])}")
                        with action_col:
                            if st.button("＋ Add", key=f"reco_rule_{item}", use_container_width=True):
                                if item not in st.session_state.basket:
                                    st.session_state.basket.append(item)
                                st.rerun()
                        st.markdown(
                            f'<div class="reco-metrics"><b>Lift {rule["lift"]:.2f}×</b>'
                            f' &nbsp; · &nbsp; Confidence {rule["confidence"]:.0%}'
                            f' &nbsp; · &nbsp; Support {rule["support"]:.3f}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("Why this recommendation?"):
                            st.write(explanation_text(rule, item))

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

    st.json(metadata)# -------------------------------------------------
# Storefront layout
# -------------------------------------------------

shop_col, cart_col = st.columns([2.2, 1], gap="large")

with shop_col:
    st.markdown('<div class="section-title">Shop the catalogue</div>', unsafe_allow_html=True)
    search_col, category_col = st.columns([1.6, 1], gap="small")
    with search_col:
        search = st.text_input("Search products", placeholder="Search by product name…")
    with category_col:
        categories = sorted(products["Category"].dropna().unique())
        category = st.selectbox("Category", ["All"] + categories)

    filtered = products.copy()
    if category != "All":
        filtered = filtered[filtered["Category"] == category]
    if search:
        mask = (
            filtered["Item"].str.contains(search, case=False, na=False, regex=False)
            | filtered["Item_EN"].str.contains(search, case=False, na=False, regex=False)
        )
        filtered = filtered[mask]

    st.markdown(
        f'<div class="muted-note">{len(filtered)} products available · Browse & add to your basket</div>',
        unsafe_allow_html=True,
    )
    show_all = st.checkbox("Show all matching products", value=False)
    visible_products = filtered if show_all else filtered.head(12)
    if not show_all and len(filtered) > 12:
        st.caption("Showing the first 12 products. Search, filter, or select Show all to explore more.")

    product_cols = st.columns(3, gap="small")
    for index, (_, product) in enumerate(visible_products.iterrows()):
        item = str(product["Item"])
        safe_item = escape(item)
        safe_en = escape(str(product["Item_EN"]))
        safe_category = escape(str(product["Category"]))
        with product_cols[index % 3]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="product-emoji">{product_icon(product["Category"])}</div>'
                    f'<div class="product-name">{safe_item}</div>'
                    f'<div class="product-category">{safe_en} · {safe_category}</div>'
                    f'<div class="product-price">${float(product["Price"]):.2f}</div>',
                    unsafe_allow_html=True,
                )
                already_added = item in st.session_state.basket
                if st.button(
                    "✓ In basket" if already_added else "＋ Add to basket",
                    key=f"catalogue_{item}", disabled=already_added,
                    use_container_width=True, type="secondary",
                ):
                    st.session_state.basket.append(item)
                    st.rerun()

with cart_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">🛒 Your basket</div>', unsafe_allow_html=True)
        basket_items = st.session_state.basket
        if not basket_items:
            st.info("Your basket is empty. Add a product to get started.")
        else:
            total = 0.0
            for item in list(basket_items):
                product = get_product(item)
                if product is None:
                    continue
                price = float(product["Price"])
                total += price
                line_col, remove_col = st.columns([3.1, 1], vertical_alignment="center")
                with line_col:
                    st.markdown(
                        f'<div class="cart-name">{escape(item)}</div>'
                        f'<div class="cart-note">${price:.2f} · {escape(str(product["Category"]))}</div>',
                        unsafe_allow_html=True,
                    )
                with remove_col:
                    if st.button("✕", key=f"remove_{item}", help=f"Remove {item}", use_container_width=True):
                        st.session_state.basket.remove(item)
                        st.rerun()
                st.divider()

            st.metric("Basket total", f"${total:.2f}")
            st.caption(f"{len(basket_items)} selected product(s)")
            if st.button("Clear basket", use_container_width=True):
                st.session_state.basket = []
                st.rerun()

st.divider()
st.markdown('<div class="section-title">✨ Picked for your basket</div>', unsafe_allow_html=True)

basket_items = st.session_state.basket
recommendations = []
fired_rules = []

if basket_items:
    recommendations, fired_rules = find_recommendations(basket_items)
    if recommendations:
        st.markdown('<div class="muted-note">Based on products purchased together · Ranked by lift</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="promo">No matching association rule for this basket — showing popular picks instead.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="muted-note">Your basket is empty · Explore popular picks to get started</div>', unsafe_allow_html=True)

if recommendations:
    display_items = [(item, get_product(item), rule) for item, rule in recommendations]
else:
    display_items = [(result["item"], result["product"], None) for result in popular_products(limit=5)]

reco_cols = st.columns(3, gap="small")
for index, (item, product, rule) in enumerate(display_items):
    if product is None:
        continue
    with reco_cols[index % 3]:
        with st.container(border=True):
            st.markdown(
                f'<div class="product-name">{product_icon(product["Category"])} '
                f'{escape(str(item))}</div>'
                f'<div class="reco-meta">{escape(str(product["Category"]))} · '
                f'${float(product["Price"]):.2f}</div>',
                unsafe_allow_html=True,
            )
            if rule is not None:
                st.markdown(
                    f'<div class="reco-stat"><b>Lift {rule["lift"]:.2f}×</b> · '
                    f'Confidence {rule["confidence"]:.0%} · '
                    f'Support {rule["support"]:.3f}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Why this product?"):
                    st.write(explanation_text(rule, item))
            else:
                st.caption("⭐ Popular choice")
            if st.button("＋ Add to basket", key=f"recommendation_{item}", use_container_width=True):
                if item not in st.session_state.basket:
                    st.session_state.basket.append(item)
                st.rerun()

if basket_items:
    with st.expander(f"🔎 View rules that fired ({len(fired_rules)})"):
        if not fired_rules:
            st.caption("No rules fired for this basket.")
        else:
            fired_rows = [
                {
                    "Antecedent": ", ".join(rule["antecedents"]),
                    "Consequent": ", ".join(rule["consequents"]),
                    "Support": rule["support"],
                    "Confidence": rule["confidence"],
                    "Lift": rule["lift"],
                }
                for rule in fired_rules
            ]
            st.dataframe(
                pd.DataFrame(fired_rows).sort_values("Lift", ascending=False),
                use_container_width=True, hide_index=True,
            )

with st.expander("ℹ️ Model information"):
    st.json(metadata)
