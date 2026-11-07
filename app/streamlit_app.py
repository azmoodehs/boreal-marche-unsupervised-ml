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

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1500px; padding-bottom: 2.5rem;}
      h1 {font-size: 2rem !important; letter-spacing: -.04em; margin-bottom: .2rem;}
      h2 {font-size: 1.1rem !important; letter-spacing: -.02em; padding: 0 !important;}
      h3 {font-size: 1rem !important; padding: 0 !important;}
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
      }
      div[data-testid="stMetric"] {padding: .1rem 0;}
      div[data-testid="stMetricLabel"] {font-size: .72rem;}
      div[data-testid="stMetricValue"] {font-size: 1.1rem;}
      .app-kicker {font-size:.78rem; font-weight:700; letter-spacing:.12em; color:#13856a;}
      .app-subtitle {color: gray; font-size:.89rem; margin-top:.1rem; margin-bottom:1.2rem;}
      .reco-name {font-weight:700; font-size:.98rem; letter-spacing:-.015em;}
      .reco-meta {font-size:.8rem; opacity:.78; margin-top:.12rem;}
      .reco-metrics {font-size:.8rem; opacity:.85; margin:.35rem 0 .5rem;}
      .reco-metrics b {color:#16816a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">BORÉAL MARCHÉ / SMART SHOPPING</div>', unsafe_allow_html=True)
st.title("Shopping recommendations")
st.markdown('<div class="app-subtitle">Discover products frequently bought together · Powered by Apriori</div>', unsafe_allow_html=True)



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

    st.json(metadata)