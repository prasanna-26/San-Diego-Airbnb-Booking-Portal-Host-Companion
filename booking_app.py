import streamlit as st
import pandas as pd

# App Configuration
st.set_page_config(page_title="San Diego Airbnb Booking", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('booking_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Please run the data preparation cell first.")
    st.stop()

# Header
st.title("🏡 San Diego Airbnb Booking Portal")
st.markdown("Explore top-rated stays, view guest sentiment, and book your perfect getaway.")

# --- Sidebar Filters ---
st.sidebar.header("Filter Stays")

# Price Range
min_p, max_p = int(df['price'].min()), int(df['price'].max())
price_range = st.sidebar.slider("Price Range ($)", min_p, max_p, (50, 500))

# Number of Guests Filter
if 'accommodates' in df.columns:
    max_guests = int(df['accommodates'].max())
    num_guests = st.sidebar.number_input("Minimum Guests", min_value=1, max_value=max_guests, value=2, step=1)
else:
    num_guests = None

# Bedrooms Filter
if 'bedrooms' in df.columns:
    max_bedrooms = int(df['bedrooms'].max())
    min_bedrooms = st.sidebar.number_input("Minimum Bedrooms", min_value=1, max_value=max_bedrooms, value=1, step=1)
else:
    min_bedrooms = None

# Beds Filter
if 'beds' in df.columns:
    max_beds = int(df['beds'].max())
    min_beds = st.sidebar.number_input("Minimum Beds", min_value=0, max_value=max_beds, value=0, step=1)
else:
    min_beds = None

# Bathrooms Filter
if 'bathrooms' in df.columns:
    max_bathrooms = int(df['bathrooms'].max())
    min_bathrooms = st.sidebar.number_input("Minimum Bathrooms", min_value=1, max_value=max_bathrooms, value=1, step=1)
else:
    min_bathrooms = None

# Sentiment Filter
min_sentiment = st.sidebar.slider("Minimum Sentiment Score", -1.0, 1.0, 0.2)

# Neighborhood Filter
neighborhoods = sorted(df['neighbourhood_cleansed'].unique().astype(str))
selected_neighborhood = st.sidebar.multiselect("Neighborhood", neighborhoods)

# Sort By Filter
sort_option = st.sidebar.selectbox("Sort By", 
    ["Relevance", "Best Value", "Highest Price", "Lowest Price"],
    index=0)

# Apply Filters
filtered_df = df[
    (df['price'] >= price_range[0]) &
    (df['price'] <= price_range[1]) &
    (df['mean_compound'] >= min_sentiment)
]

# Apply optional filters only if columns exist and filter values are set
if num_guests is not None and 'accommodates' in df.columns:
    filtered_df = filtered_df[filtered_df['accommodates'] >= num_guests]

if min_bedrooms is not None and 'bedrooms' in df.columns:
    filtered_df = filtered_df[filtered_df['bedrooms'] >= min_bedrooms]

if min_beds is not None and 'beds' in df.columns:
    filtered_df = filtered_df[filtered_df['beds'] >= min_beds]

if min_bathrooms is not None and 'bathrooms' in df.columns:
    filtered_df = filtered_df[filtered_df['bathrooms'] >= min_bathrooms]

if selected_neighborhood:
    filtered_df = filtered_df[filtered_df['neighbourhood_cleansed'].isin(selected_neighborhood)]

# Apply Sorting
if sort_option == "Lowest Price":
    filtered_df = filtered_df.sort_values('price', ascending=True)
elif sort_option == "Highest Price":
    filtered_df = filtered_df.sort_values('price', ascending=False)
elif sort_option == "Best Value":
    # Best value = good sentiment + reasonable price (lower price/sentiment ratio is better)
    filtered_df['value_score'] = filtered_df['price'] / (filtered_df['mean_compound'] + 1.5)  # Add 1.5 to avoid division issues
    filtered_df = filtered_df.sort_values('value_score', ascending=True)
    filtered_df = filtered_df.drop(columns=['value_score'])
# else: Relevance (default order, no sorting)

st.markdown(f"**Found {len(filtered_df)} listings matching your criteria.**")

# --- Main Content ---

# Layout: List of properties
# Display up to 50 listings
display_df = filtered_df.head(50)

if len(display_df) == 0:
    st.warning("No listings found matching your filters. Try adjusting your criteria.")
else:
    for index, row in display_df.iterrows():
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if pd.notna(row['picture_url']):
                    st.image(row['picture_url'], use_container_width=True)
                else:
                    st.write("No Image Available")
            
            with col2:
                st.subheader(row['name'])
                st.caption(f"{row['neighbourhood_cleansed']} • {row['room_type']}")
                
                # Details row
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"👥 **{row['accommodates']}** Guests")
                c2.markdown(f"🛏️ **{row['bedrooms']}** Bedrooms")
                c3.markdown(f"🛁 **{row['bathrooms']}** Bathrooms")
                
                # Sentiment Display
                score = row['mean_compound']
                if score >= 0.5: 
                    sent_str = "🌟 Excellent"
                    color = "green"
                elif score >= 0.05: 
                    sent_str = "🙂 Good"
                    color = "blue"
                else: 
                    sent_str = "😐 Mixed/Neutral"
                    color = "orange"
                
                st.markdown(f"**Guest Sentiment:** :{color}[{sent_str}] ({score:.2f}) based on {row['n_reviews']} reviews")
                
                # Price & Booking Action
                c_price, c_btn = st.columns([1, 2])
                with c_price:
                    st.markdown(f"### ${row['price']:.0f} / night")
                with c_btn:
                    # Unique key for each button
                    if st.button(f"Select {row['name'][:20]}...", key=f"btn_{row['id']}"):
                        st.session_state['selected_id'] = row['id']
                        st.rerun()

# --- Checkout Section ---
if 'selected_id' in st.session_state:
    # Find the selected row
    selected_row = df[df['id'] == st.session_state['selected_id']].iloc[0]
    
    st.markdown("---")
    st.markdown("## 🛒 Booking Checkout")
    
    with st.form("checkout_form"):
        st.info(f"Booking: **{selected_row['name']}**")
        
        col_a, col_b = st.columns(2)
        with col_a:
             st.image(selected_row['picture_url'], width=300)
        with col_b:
             nights = st.number_input("Number of Nights", min_value=1, value=3)
             service_fee = 50
             total_price = (selected_row['price'] * nights) + service_fee
             
             st.write(f"Price per night: ${selected_row['price']:.0f}")
             st.write(f"Service Fee: ${service_fee}")
             st.markdown(f"### Total: ${total_price:.0f}")
        
        # Checkout Actions
        submitted = st.form_submit_button("Confirm & Pay")
        
        if submitted:
            st.balloons()
            st.success("Booking Confirmed! Thank you for using our portal.")
            st.markdown(f"[Link to original listing on Airbnb]({selected_row['listing_url']})")
            
    if st.button("Cancel / Close Checkout"):
        del st.session_state['selected_id']
        st.rerun()