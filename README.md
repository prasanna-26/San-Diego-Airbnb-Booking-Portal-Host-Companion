# 🏠 San Diego Airbnb Analytics & Applications

A comprehensive analytics project that combines machine learning and natural language processing to understand Airbnb pricing dynamics in San Diego, delivered through two interactive web applications.

---

## 📊 Project Overview

This project transforms raw Airbnb listing data and guest reviews into actionable insights through two complementary applications:

1. **Host Companion App** - Price prediction and market analysis tool for Airbnb hosts
2. **Booking Portal App** - Smart search and booking interface for guests

Built on a foundation of 12,900+ listings and 850,000+ guest reviews, these applications demonstrate the practical application of data science in the short-term rental market.

---

## 🎯 What Problem Does This Solve?

### For Hosts:
- **Pricing Uncertainty**: "Am I charging too much or leaving money on the table?"
- **Market Positioning**: "How does my listing compare to competitors?"
- **Guest Sentiment**: "What do guests really think about my property?"

### For Guests:
- **Information Overload**: "Which listing offers the best value?"
- **Trust Issues**: "Can I trust the rating, or should I read all reviews?"
- **Search Efficiency**: "How do I filter 12,900 listings to find the right one?"

---

## 🚀 Applications

### 1. Host Companion App (`app.py`)

**Purpose**: Empower Airbnb hosts with data-driven pricing recommendations and market insights.

#### Features:

**🔮 Price Prediction**
- Neural network-powered nightly rate estimation
- Considers 20+ features including property details, location, and host metrics
- Average prediction error: ~$58/night
- Explains 60% of price variation in the San Diego market

**💬 Sentiment Analysis**
- Analyzes guest reviews using NLP (NLTK VADER)
- Provides sentiment breakdown: Positive, Neutral, Negative percentages
- Shows overall sentiment score (-1 to +1 scale)
- Helps hosts understand guest satisfaction beyond star ratings

**📈 Market Insights**
- Interactive price distribution across San Diego neighborhoods
- Neighborhood-level pricing trends
- Visual comparison of your listing vs. market averages
- Price vs. Rating scatter plots for competitive positioning

**📍 Location Intelligence**
- Heatmap showing price variations across the city
- Identifies premium neighborhoods
- Micro-location effects on pricing

#### How It Works:

![Host App Workflow](images/host_app_workflow.png)

1. **Input Property Details**: Enter bedrooms, bathrooms, beds, guest capacity
2. **Add Location**: Provide coordinates or select neighborhood
3. **Host Information**: Superhost status, response rate, acceptance rate
4. **Review Metrics**: Number of reviews and review scores
5. **Get Predictions**: Receive instant price estimate and market insights
6. **Analyze Sentiment**: See what guests are saying about similar properties

#### What The Model Considers:

**Property Features:**
- Number of bedrooms, bathrooms, beds
- Maximum guest capacity
- Room type (Entire home, Private room, Shared room)

**Location Data:**
- Latitude and longitude coordinates
- Neighborhood classification
- Distance to key attractions (beaches, downtown)

**Host Metrics:**
- Superhost status (TRUE/FALSE)
- Host response rate (0-100%)
- Host acceptance rate (0-100%)

**Guest Feedback:**
- Total number of reviews
- Review frequency
- Sentiment scores from review text

#### Technical Stack:

- **Model**: TensorFlow/Keras Neural Network (5 layers, dropout regularization)
- **Scaling**: StandardScaler for feature normalization
- **NLP**: NLTK VADER for sentiment analysis
- **Visualization**: Plotly for interactive charts
- **Framework**: Streamlit for web interface

#### Screenshots:

**Main Interface - Price Prediction**
![Price Prediction Interface](images/app_price_prediction.png)

**Sentiment Analysis Dashboard**
![Sentiment Analysis](images/app_sentiment_analysis.png)

**Market Insights - Neighborhood Comparison**
![Market Insights](images/app_market_insights.png)

**Price Distribution Visualization**
![Price Distribution](images/app_price_distribution.png)

---

### 2. Booking Portal App (`booking_app.py`)

**Purpose**: Provide guests with an intelligent search and booking interface for San Diego Airbnb listings.

#### Features:

**🔍 Smart Filtering**
- **Price Range**: Interactive slider ($0 - $5000+)
- **Guest Capacity**: Minimum guests filter (shows properties that fit your group or larger)
- **Bedrooms**: Minimum 1-10+ bedrooms
- **Bathrooms**: Minimum 1-8+ bathrooms
- **Neighborhood**: Multi-select neighborhood filter
- **Sentiment Score**: Filter by guest satisfaction level

**📊 Intelligent Sorting**
- **Relevance**: Default ordering (best match)
- **Best Value**: Optimized price-to-quality ratio (price / sentiment score)
- **Lowest Price**: Budget-conscious travelers
- **Highest Price**: Luxury seekers

**⭐ Guest Sentiment Display**
- Color-coded sentiment indicators:
  - 🌟 **Excellent** (score ≥ 0.5) - Green
  - 🙂 **Good** (score ≥ 0.05) - Blue
  - 😐 **Mixed/Neutral** (score < 0.05) - Orange
- Based on sentiment analysis of actual guest reviews
- Shows number of reviews for confidence assessment

**🛒 Booking Flow**
- One-click property selection
- Checkout simulation with:
  - Property image and details
  - Nightly rate calculation
  - Service fee ($50)
  - Total cost breakdown
- Link to original Airbnb listing

---

#### How It Works:
1. **Set Filters**: Define your requirements (guests, price, location, etc.)
2. **Browse Listings**: View up to 50 matching properties
3. **Review Details**: See photos, sentiment scores, and key features
4. **Compare Options**: Use sorting to find best value or best price
5. **Select Property**: Click "Select" button on preferred listing
6. **Checkout**: Review booking details and total cost
7. **Confirm**: Complete booking simulation

[Booking Portal Workflow] 
<img width="958" height="541" alt="image" src="https://github.com/user-attachments/assets/6bfb6edb-9817-4b9e-9d1b-c90dd974eb3f" />

#### What It Displays:

**For Each Listing:**
- Property name and high-quality image
- Neighborhood and room type
- Guest capacity, bedrooms, bathrooms
- Guest sentiment (Excellent/Good/Mixed) with score
- Number of reviews for reliability
- Price per night
- Quick-select booking button

**Search Results:**
- Total listings found matching criteria
- Top 50 results displayed
- Warning message if no matches found

#### Best Value Calculation:

The "Best Value" sorting uses this formula:

```
value_score = price / (sentiment_score + 1.5)
```

**Lower score = Better value**

This balances:
- Affordability (lower price)
- Quality (higher sentiment)
- Prevents division by zero with +1.5 offset

Example:
- Listing A: $200/night, sentiment 0.8 → score: 86.96
- Listing B: $150/night, sentiment 0.5 → score: 75.00 ✅ Better Value

#### Screenshots:

**Main Search Interface**
![Booking Portal Home](images/booking_home.png)

**Filter Panel**
![Filter Options](images/booking_filters.png)

**Listing Display with Sentiment**
![Listing Cards](images/booking_listings.png)

**Checkout Flow**
![Checkout Interface](images/booking_checkout.png)

---

## 📁 Project Structure

```
AirBNB app/
│
├── app.py                          # Host Companion application
├── booking_app.py                  # Booking Portal application
│
├── best_model.keras                # Trained neural network model
├── scaler.joblib                   # StandardScaler for feature normalization
├── feature_names.json              # Model input feature list
│
├── listings.csv                    # Core listing data (~12.9K listings)
├── listings_full.csv               # Extended listing data with descriptions
├── booking_data.csv                # Processed data for booking portal
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── project_applications_overview.txt  # Project summary and LinkedIn post
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM recommended
- Internet connection (for initial package downloads)

### Step 1: Clone or Download Project

```bash
cd "C:\Users\YourUsername\Downloads\AirBNB app"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.14.0
scikit-learn>=1.3.0
nltk>=3.8.0
plotly>=5.17.0
joblib>=1.3.0
```

### Step 3: Download NLTK Data (First Time Only)

Open Python and run:

```python
import nltk
nltk.download('vader_lexicon')
```

---

## ▶️ Running the Applications

### Option 1: Host Companion App

**Start the app:**
```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Access the app:**
- Open your browser
- Navigate to `http://localhost:8501`
- Start entering property details

**To stop:**
- Press `Ctrl+C` in the terminal

---

### Option 2: Booking Portal App

**Start the app:**
```bash
streamlit run booking_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Access the app:**
- Open your browser
- Navigate to `http://localhost:8501`
- Set your filters and browse listings

**To stop:**
- Press `Ctrl+C` in the terminal

---

## 📖 Usage Examples

### Example 1: Host Pricing Analysis

**Scenario**: You have a 2-bedroom beachfront condo in Pacific Beach and want to know the optimal nightly rate.

**Steps:**
1. Run `streamlit run app.py`
2. Enter property details:
   - Bedrooms: 2
   - Bathrooms: 2
   - Beds: 2
   - Accommodates: 4 guests
   - Latitude: 32.7941
   - Longitude: -117.2523
   - Superhost: Yes
   - Response Rate: 95%
3. Click "Estimate Nightly Price"
4. **Result**: Predicted price: ~$280/night
5. View sentiment analysis to understand guest expectations
6. Check market insights to see Pacific Beach pricing trends

![Example Host Analysis](images/example_host_analysis.png)

---

### Example 2: Guest Search for Family Vacation

**Scenario**: Family of 6 looking for a vacation rental under $400/night near the beach.

**Steps:**
1. Run `streamlit run booking_app.py`
2. Set filters:
   - Price Range: $100 - $400
   - Minimum Guests: 6
   - Minimum Bedrooms: 3
   - Minimum Bathrooms: 2
   - Neighborhood: Pacific Beach, Mission Beach
   - Sort By: Best Value
3. Browse results (Found 17 listings)
4. Select listing with "Excellent" sentiment
5. Review checkout details
6. **Result**: Found 3BR/2BA property at $325/night with 0.78 sentiment score

![Example Guest Search](images/example_guest_search.png)

---

## 📊 Data Overview

### listings.csv
**Size**: ~12,900 rows, 75 columns

**Key Columns:**
- `id`: Unique listing identifier
- `name`: Property name
- `price`: Nightly rate
- `latitude`, `longitude`: Coordinates
- `neighbourhood_cleansed`: Neighborhood name
- `room_type`: Entire home/Private room/Shared room
- `accommodates`: Guest capacity
- `bedrooms`, `bathrooms`, `beds`: Property specs
- `host_is_superhost`: Superhost status
- `review_scores_rating`: Average rating (0-5)
- `number_of_reviews`: Total review count

### listings_full.csv
Extended version with:
- Property descriptions
- Amenity lists
- House rules
- Host information

### booking_data.csv
**Size**: ~12,900 rows, 20 columns

**Processed for booking portal with:**
- `mean_compound`: Sentiment score from reviews
- `n_reviews`: Review count
- `picture_url`: Property image URL
- Filtered and cleaned data

---

## 🧮 Model Details

### Neural Network Architecture

```python
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Params
=================================================================
Dense (128 neurons)         (None, 128)              2,688
Dropout (0.3)               (None, 128)              0
Dense (64 neurons)          (None, 64)               8,256
Dropout (0.2)               (None, 64)               0
Dense (32 neurons)          (None, 32)               2,080
Dense (1 neuron)            (None, 1)                33
=================================================================
Total params: 13,057
Trainable params: 13,057
```

### Performance Metrics

- **R² Score**: ~0.60 (explains 60% of price variance)
- **Mean Absolute Error (MAE)**: ~$58 per night
- **Root Mean Squared Error (RMSE)**: ~$95 per night
- **Training Data**: 10,320 listings (80% split)
- **Validation Data**: 2,580 listings (20% split)

### Feature Importance (Top 10)

1. **accommodates** - Guest capacity
2. **bedrooms** - Number of bedrooms
3. **bathrooms** - Number of bathrooms
4. **latitude** - North-south position (beach proximity)
5. **longitude** - East-west position
6. **room_type** - Entire home vs. private/shared room
7. **host_is_superhost** - Superhost premium
8. **number_of_reviews** - Social proof
9. **review_scores_rating** - Overall rating
10. **beds** - Total beds available

---

## 📚 Key Learnings

### Technical Insights

1. **Feature Engineering Matters More Than Model Complexity**
   - Property size features (bedrooms, beds) explained more variance than complex interactions
   - Location coordinates captured micro-location effects better than neighborhood categories

2. **NLP Adds Real Value**
   - Sentiment analysis from reviews provided insights beyond star ratings
   - Guest language reveals priorities: cleanliness, communication, location

3. **User Experience Trumps Model Accuracy**
   - Simple filters (price, bedrooms) matter more than perfect predictions
   - Users want explanations, not just numbers

4. **Defensive Programming Is Critical**
   - Column existence checks prevented runtime errors
   - Graceful degradation when data is missing

### Product Insights

1. **Hosts Need Context, Not Just Predictions**
   - "Your price should be $280" is less useful than
   - "You're 15% above market average for Pacific Beach 2BR condos"

2. **Guests Want Filters + Trust Signals**
   - Price and location are table stakes
   - Sentiment scores build trust more than star ratings

3. **The Gap Between Prototype and Production Is Real**
   - Model works in Jupyter ≠ Model works in production
   - "Works on my machine" is not a deployment strategy

---

**Data Source:**
- San Diego's listing data :- https://data.insideairbnb.com/united-states/ca/san-diego/2025-06-21/data/listings.csv.gz) - 
- Airbnb public review data: (850,000+ reviews) :- https://data.insideairbnb.com/united-states/ca/san-diego/2025-06-21/data/reviews.csv.gz

**Tools & Technologies:**
- [Streamlit](https://streamlit.io/) - Web application framework
- [TensorFlow](https://www.tensorflow.org/) - Machine learning model
- [NLTK](https://www.nltk.org/) - Natural language processing
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation

---

## 📞 Contact & Links

**Previous Work:**
- [Project Presentation (PPT)]()
- [Google Colab Notebook](https://drive.google.com/file/d/174GQDYW5819cOEDlon6UCKCJXT_IyYBl/view?usp=sharing)

**Connect:**
- LinkedIn: [Your LinkedIn Profile]
- Email: [prasannajain265@gmail.com]
- GitHub: [Your GitHub Profile]

## 📄 License

This project is developed for educational and portfolio purposes. 

**Data Usage**: Follows Inside Airbnb's Creative Commons License for non-commercial use.

---

## 🙏 Final Notes

This project represents my first serious attempt at building production-ready applications in 6-7 years. It's been a humbling journey involving:

- Learning Streamlit state management the hard way
- Realizing "it works in Colab" ≠ "it's production ready"
- Understanding that users want filters, not feature importance plots

**Key Takeaway**: Good analytics without usability is just research. The real value comes from making insights accessible and actionable.

If you're building something similar or have suggestions for improvement, I'd love to hear from you!

*Built with ❤️ and lots of coffee ☕ in San Diego, CA*
