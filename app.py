import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import html as _html

# Page config
st.set_page_config(page_title="San Diego Airbnb Host Companion", layout="wide")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # Download VADER lexicon if not present
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    # Load model and scaler
    model = tf.keras.models.load_model('best_model.keras')
    scaler = joblib.load('scaler.joblib')

    # Load feature names
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)

    # Load dataset for insights
    df_listings = pd.read_csv('listings.csv')
    # Load full listing file (contains name, url, id etc.) if present.
    # Try common filenames: 'listing_full.csv' and 'listings_full.csv'.
    df_full = pd.DataFrame()
    for fname in ('listing_full.csv', 'listings_full.csv'):
        try:
            df_full = pd.read_csv(fname)
            break
        except Exception:
            df_full = pd.DataFrame()

    return model, scaler, feature_names, df_listings, df_full

model, scaler, feature_names, df_listings, df_full = load_assets()

# --- Title ---
st.title("San Diego Airbnb Host Companion")
st.markdown("This app helps hosts estimate pricing, analyze review sentiment, and explore market trends.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Price Prediction", "Sentiment Analysis", "Market Insights"])

# ==========================
# TAB 1: Price Prediction
# ==========================
with tab1:
    st.header("Estimate Nightly Price")
    st.write("Enter listing details below to predict the nightly price based on our neural network model.")

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Features: ['accommodates', 'bedrooms', 'bathrooms', 'beds', 'latitude', 'longitude',
        #            'host_is_superhost', 'host_response_rate', 'host_listings_count',
        #            'review_scores_rating', 'number_of_reviews', 'reviews_per_month',
        #            'mean_compound', 'n_reviews']

        with col1:
            accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=4)
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
            beds = st.number_input("Beds", min_value=0, max_value=20, value=2)

        with col2:
            # Defaults near San Diego center
            latitude = st.number_input("Latitude", value=32.7157)
            longitude = st.number_input("Longitude", value=-117.1611)
            host_is_superhost = st.selectbox("Is Superhost?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            host_response_rate = st.slider("Host Response Rate", 0.0, 1.0, 0.95)
            host_listings_count = st.number_input("Host Total Listings", min_value=1, value=1)

        with col3:
            review_scores_rating = st.slider("Review Rating (0-5)", 0.0, 5.0, 4.8)
            number_of_reviews = st.number_input("Total Reviews", min_value=0, value=10, step=1)
            reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=1.0)
            mean_compound = st.slider("Avg Sentiment Score (-1 to 1)", -1.0, 1.0, 0.5)
            n_reviews = st.number_input("Number of Reviews (Sentiment)", min_value=0, value=10, step=1)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # Construct input array in correct order
        input_data = np.array([[
            accommodates, bedrooms, bathrooms, beds,
            latitude, longitude,
            host_is_superhost, host_response_rate, host_listings_count,
            review_scores_rating, number_of_reviews, reviews_per_month,
            mean_compound, n_reviews
        ]])

        # Scale inputs
        input_scaled = scaler.transform(input_data)

        # Predict log price
        pred_log = model.predict(input_scaled)

        # Inverse log transform (expm1)
        pred_price = np.expm1(pred_log)[0][0]

        st.success(f"Predicted Nightly Price: **${pred_price:.2f}**")

        # Show nearby listings by price: find 10 listings with price closest to prediction
        try:
            df_search = df_listings.copy()

            # normalize price column to numeric if necessary
            if 'price' in df_search.columns and not np.issubdtype(df_search['price'].dtype, np.number):
                def _parse_price(x):
                    try:
                        return float(str(x).replace('$', '').replace(',', ''))
                    except Exception:
                        return np.nan
                df_search['price'] = df_search['price'].apply(_parse_price)

            # drop rows without price
            df_search = df_search[df_search['price'].notna()].copy()

            # compute absolute difference and pick nearest
            df_search['price_diff'] = (df_search['price'] - pred_price).abs()
            nearby = df_search.nsmallest(10, 'price_diff')

            # Build display rows: name/id, avg rating, number of reviews, price
            # If we have a full listing file, try to merge to get `name` and `listing_url`.
            # Be robust to different id column names and types.
            def find_common_id(col_candidates, a_cols, b_cols):
                a_low = {c.lower(): c for c in a_cols}
                b_low = {c.lower(): c for c in b_cols}
                for cand in col_candidates:
                    if cand in a_low and cand in b_low:
                        return a_low[cand], b_low[cand]
                return None, None

            rows = []
            if not df_full.empty:
                # candidate id names to try
                candidates = ['id', 'listing_id', 'listingid', 'listing id', 'property_id']
                a_col, b_col = find_common_id(candidates, nearby.columns.tolist(), df_full.columns.tolist())
                if a_col and b_col:
                    # merge on found id columns
                    nearby[a_col] = nearby[a_col].astype(str)
                    df_full[b_col] = df_full[b_col].astype(str)
                    nearby = nearby.merge(df_full[[b_col, 'name', 'listing_url']], left_on=a_col, right_on=b_col, how='left')
                else:
                    # no common id column found — attempt per-row lookup in df_full by matching price & reviews
                    # prepare df_full numeric price column if present
                    if 'price' in df_full.columns and not np.issubdtype(df_full['price'].dtype, np.number):
                        def _parse_price(x):
                            try:
                                return float(str(x).replace('$','').replace(',',''))
                            except Exception:
                                return np.nan
                        df_full['price_parsed'] = df_full['price'].apply(_parse_price)
                    else:
                        df_full['price_parsed'] = df_full['price'] if 'price' in df_full.columns else np.nan

            # For each nearby row, build display fields using merged info when available, else attempt lookup
            for _, r in nearby.iterrows():
                title = None
                listing_url = None
                # if merge produced a name column, use it
                if 'name' in r.index and pd.notna(r.get('name')):
                    title = str(r.get('name'))
                    listing_url = r.get('listing_url') if 'listing_url' in r.index else None
                else:
                    # try to find matching row in df_full by id variants
                    matched = None
                    if not df_full.empty:
                        # try id-based match
                        for cand in ['id','listing_id','listingid','property_id']:
                            if cand in nearby.columns and cand in df_full.columns:
                                key = str(r.get(cand))
                                tmp = df_full[df_full[cand].astype(str) == key]
                                if not tmp.empty:
                                    matched = tmp.iloc[0]
                                    break
                        # try match by price and number_of_reviews
                        if matched is None and 'price' in r.index and 'number_of_reviews' in r.index:
                            price_val = r.get('price')
                            numrev = r.get('number_of_reviews')
                            if pd.notna(price_val) and pd.notna(numrev):
                                cand_rows = df_full[(df_full.get('price_parsed', np.nan) == price_val) & (df_full.get('number_of_reviews') == numrev)]
                                if not cand_rows.empty:
                                    matched = cand_rows.iloc[0]
                    if matched is not None:
                        title = matched.get('name') if 'name' in matched.index else None
                        listing_url = matched.get('listing_url') if 'listing_url' in matched.index else None

                if not title:
                    if 'id' in r.index and pd.notna(r.get('id')):
                        try:
                            title = f"Listing {int(r.get('id'))}"
                        except Exception:
                            title = f"Listing"
                    else:
                        title = f"Listing"

                avg_rating = r.get('review_scores_rating') if 'review_scores_rating' in r.index else None
                n_reviews = r.get('number_of_reviews') if 'number_of_reviews' in r.index else None
                price_disp = f"${r.get('price'):.2f}" if pd.notna(r.get('price')) else 'N/A'

                # listing column: name before price
                listing_col = f"{title} — {price_disp}"

                rows.append({
                    'listing': listing_col,
                    'avg_rating': f"{avg_rating:.2f}" if pd.notna(avg_rating) else 'N/A',
                    'n_reviews': int(n_reviews) if pd.notna(n_reviews) else 'N/A',
                    'price': price_disp,
                    'link': listing_url if pd.notna(listing_url) else 'N/A'
                })

            if rows:
                st.subheader("Nearby Listings")
                st.write(f"Listings closest to the predicted nightly price of ${pred_price:.2f}:")
                # Render an HTML table with columns: Price, Listing (clickable), Avg Rating, Reviews
                # Build table rows
                table_rows = []
                for r in rows:
                    title = r.get('listing', 'Listing')
                    link = r.get('link', None)
                    avg = r.get('avg_rating', 'N/A')
                    nrev = r.get('n_reviews', 'N/A')
                    price_disp = r.get('price', 'N/A')

                    safe_title = _html.escape(title)
                    if link and link != 'N/A':
                        listing_html = f'<a href="{_html.escape(link)}" target="_blank">{safe_title}</a>'
                    else:
                        listing_html = safe_title

                    table_rows.append((price_disp, listing_html, avg, nrev))

                # build HTML table
                table_html = ['<table style="width:100%; border-collapse:collapse">',
                              '<thead><tr>',
                              '<th style="text-align:left; padding:8px; border-bottom:1px solid #444">Price</th>',
                              '<th style="text-align:left; padding:8px; border-bottom:1px solid #444">Listing</th>',
                              '<th style="text-align:left; padding:8px; border-bottom:1px solid #444">Avg Rating</th>',
                              '<th style="text-align:left; padding:8px; border-bottom:1px solid #444">Reviews</th>',
                              '</tr></thead>',
                              '<tbody>']
                for price_disp, listing_html, avg, nrev in table_rows:
                    table_html.append('<tr>')
                    table_html.append(f'<td style="padding:8px; border-bottom:1px solid #222">{_html.escape(str(price_disp))}</td>')
                    table_html.append(f'<td style="padding:8px; border-bottom:1px solid #222">{listing_html}</td>')
                    table_html.append(f'<td style="padding:8px; border-bottom:1px solid #222">{_html.escape(str(avg))}</td>')
                    table_html.append(f'<td style="padding:8px; border-bottom:1px solid #222">{_html.escape(str(nrev))}</td>')
                    table_html.append('</tr>')
                table_html.append('</tbody></table>')

                st.markdown('\n'.join(table_html), unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not compute nearby listings: {e}")

# ==========================
# TAB 2: Sentiment Analysis
# ==========================
with tab2:
    st.header("Analyze Guest Review")
    st.write("Enter a review text to see its sentiment score.")

    review_text = st.text_area("Paste review here:", "The place was amazing and very clean. We loved the location!")
    analyze_btn = st.button("Analyze Sentiment")

    if analyze_btn and review_text:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(review_text)
        compound = scores['compound']

        if compound >= 0.05:
            label = "Positive"
            color = "green"
        elif compound <= -0.05:
            label = "Negative"
            color = "red"
        else:
            label = "Neutral"
            color = "gray"

        st.markdown(f"### Sentiment: :{color}[{label}]")
        st.metric("Compound Score", f"{compound:.4f}")
        st.json(scores)

# ==========================
# TAB 3: Market Insights
# ==========================
with tab3:
    st.header("Market Overview")

    # 1. Price Distribution
    st.subheader("Price Distribution")
    fig_hist = px.histogram(df_listings, x='price', nbins=50, title="Distribution of Nightly Prices", color_discrete_sequence=['teal'])
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Price vs Reviews
    st.subheader("Price vs. Popularity")
    # ensure a categorical `room_type` exists for coloring (recover from one-hot if necessary)
    if 'room_type' not in df_listings.columns:
        room_cols = [c for c in df_listings.columns if c.startswith('room_type_')]
        if room_cols:
            df_listings['room_type'] = df_listings[room_cols].idxmax(axis=1).str.replace('room_type_', '', regex=False)
        else:
            df_listings['room_type'] = 'Unknown'
    fig_scatter = px.scatter(
        df_listings, x='number_of_reviews', y='price',
        color='room_type',
        title="Price vs. Number of Reviews"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Map
    st.subheader("Listing Locations")
    st.write("Listings colored by price (lighter is more expensive).")
    fig_map = px.scatter_mapbox(
        df_listings, lat="latitude", lon="longitude", color="price",
        size="accommodates", color_continuous_scale=px.colors.cyclical.IceFire,
        zoom=10, mapbox_style="open-street-map", title="Map of Listings"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# Instructions footer
st.sidebar.info("To run this app: `streamlit run app.py`")