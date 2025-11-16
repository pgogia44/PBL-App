import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Water Enhancement Market Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e8eef5;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e7bcf;
        color: white;
    }
    h1 {color: #1e3a8a;}
    h2 {color: #2563eb;}
    h3 {color: #3b82f6;}
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💧 Water Enhancement Product - Market Analytics Dashboard")
st.markdown("### Data-Driven Insights for Strategic Marketing Decisions")
st.markdown("---")

# Helper functions
def safe_column_check(df, column_name):
    """Check if column exists in dataframe"""
    return column_name in df.columns

def get_available_columns(df, column_list):
    """Return only columns that exist in the dataframe"""
    return [col for col in column_list if col in df.columns]

def create_income_mapping():
    return {
        'Less than $25,000': 1, '$25,000 - $49,999': 2, '$50,000 - $74,999': 3,
        '$75,000 - $99,999': 4, '$100,000 - $149,999': 5,
        '$150,000 - $199,999': 6, '$200,000+': 7
    }

def create_spending_mapping():
    return {
        'Less than $20': 10, '$20-$39': 30, '$40-$59': 50,
        '$60-$79': 70, '$80-$99': 90, '$100+': 120
    }

def create_usage_mapping():
    return {
        '1-2': 1.5, '3-5': 4, '6-10': 8, '11-15': 13, '16-20': 18, '21+': 25
    }

# Load data function
@st.cache_data
def load_data():
    """Load the survey data"""
    try:
        df = pd.read_csv('water_enhancement_survey_data.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please upload 'water_enhancement_survey_data.csv' to the repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load data
df = load_data()

if df is not None:
    # Display available columns for debugging
    st.sidebar.markdown("### 🔍 Dataset Info")
    with st.sidebar.expander("View Available Columns"):
        st.write(f"**Total columns:** {len(df.columns)}")
        st.write(f"**Total rows:** {len(df)}")
        for col in df.columns[:30]:
            st.text(f"• {col}")
        if len(df.columns) > 30:
            st.text(f"... and {len(df.columns)-30} more")
    
    # Create necessary mappings and columns at the start
    try:
        # Income mapping
        if safe_column_check(df, 'income'):
            income_mapping = create_income_mapping()
            df['income_numeric'] = df['income'].map(income_mapping).fillna(3)
        else:
            df['income_numeric'] = 3
        
        # Spending mapping
        if safe_column_check(df, 'monthly_beverage_spend'):
            spending_mapping = create_spending_mapping()
            df['spending_numeric'] = df['monthly_beverage_spend'].map(spending_mapping).fillna(50)
        else:
            df['spending_numeric'] = 50
        
        # Usage mapping
        if safe_column_check(df, 'weekly_usage'):
            usage_mapping = create_usage_mapping()
            df['usage_numeric'] = df['weekly_usage'].map(usage_mapping).fillna(4)
        else:
            df['usage_numeric'] = 4
        
        # Calculate monthly revenue
        if safe_column_check(df, 'willingness_to_pay_continuous'):
            df['monthly_revenue'] = df['willingness_to_pay_continuous'] * df['usage_numeric'] * 4.33
        else:
            df['monthly_revenue'] = 0
        
        # Interest level check
        if not safe_column_check(df, 'interest_level'):
            df['interest_level'] = 3
        
        # Create response_id if not exists
        if not safe_column_check(df, 'response_id'):
            df['response_id'] = range(len(df))
            
    except Exception as e:
        st.error(f"Error in data preprocessing: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/2e7bcf/ffffff?text=Water+Enhancement", use_column_width=True)
        st.markdown("## 📊 Dashboard Navigation")
        st.markdown("---")
        
        st.markdown("### Dataset Overview")
        st.metric("Total Responses", f"{len(df):,}")
        st.metric("Total Features", len(df.columns))
        
        if safe_column_check(df, 'interest_level'):
            high_interest = (df['interest_level'] >= 4).sum()
            st.metric("High Interest Customers", f"{high_interest:,} ({high_interest/len(df)*100:.1f}%)")
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        
        if safe_column_check(df, 'willingness_to_pay_continuous'):
            st.write(f"**Avg WTP:** ${df['willingness_to_pay_continuous'].mean():.2f}")
        
        if safe_column_check(df, 'purchase_likelihood'):
            likely_buyers = df['purchase_likelihood'].isin(['Definitely would purchase', 'Probably would purchase']).sum()
            st.write(f"**Likely Buyers:** {likely_buyers:,}")
        
        st.markdown("---")
        st.info("💡 **Tip:** Use tabs above to navigate between different analyses")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Marketing Insights",
        "🤖 ML Algorithms",
        "🔮 Prediction Tool",
        "📊 Association Rules"
    ])

    # ========================================================================
    # TAB 1: MARKETING INSIGHTS DASHBOARD
    # ========================================================================
    with tab1:
        st.header("📊 Strategic Marketing Insights for Customer Conversion")
        st.markdown("#### 5 Key Charts to Drive Marketing Strategy")
        
        try:
            # CHART 1: Customer Segmentation Matrix
            st.markdown("---")
            st.subheader("1️⃣ Customer Segmentation Matrix: Interest vs Income vs Age")
            st.markdown("**Action:** Target high-value segments with premium messaging")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                required_cols = ['interest_level', 'age_group', 'income']
                if all(safe_column_check(df, col) for col in required_cols):
                    df_seg = df.copy()
                    df_seg['interest_category'] = df_seg['interest_level'].apply(
                        lambda x: '🔥 High Interest' if x >= 4 else '⚡ Medium Interest' if x == 3 else '❄️ Low Interest'
                    )
                    
                    segment_data = df_seg.groupby(['age_group', 'income', 'interest_category']).agg({
                        'response_id': 'count',
                        'willingness_to_pay_continuous': 'mean' if safe_column_check(df, 'willingness_to_pay_continuous') else lambda x: 1.5
                    }).reset_index()
                    segment_data.columns = ['age_group', 'income', 'interest_category', 'count', 'avg_wtp']
                    segment_data['income_numeric'] = segment_data['income'].map(create_income_mapping())
                    
                    fig1 = px.scatter(
                        segment_data,
                        x='age_group',
                        y='income_numeric',
                        size='count',
                        color='interest_category',
                        hover_data=['income', 'count', 'avg_wtp'],
                        title='Customer Segments by Age, Income, and Interest Level',
                        color_discrete_map={
                            '🔥 High Interest': '#27ae60',
                            '⚡ Medium Interest': '#f39c12',
                            '❄️ Low Interest': '#e74c3c'
                        },
                        size_max=60,
                        height=500
                    )
                    fig1.update_yaxes(
                        tickmode='array',
                        tickvals=list(range(1, 8)),
                        ticktext=['<$25K', '$25-50K', '$50-75K', '$75-100K', '$100-150K', '$150-200K', '$200K+']
                    )
                    fig1.update_layout(template='plotly_white')
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning("⚠️ Required columns (interest_level, age_group, income) not found for segmentation chart")
            
            with col2:
                st.markdown("#### 🎯 Key Takeaways")
                if safe_column_check(df, 'age_group') and safe_column_check(df, 'interest_level'):
                    best_segment = df.groupby('age_group').apply(
                        lambda x: (x['interest_level'] >= 4).mean()
                    ).sort_values(ascending=False)
                    if len(best_segment) > 0:
                        st.success(f"**Best Segment:**\n\n{best_segment.index[0]}\n\n{best_segment.iloc[0]*100:.1f}% High Interest")
                    
                    high_income_interest = df[df['income_numeric'] >= 5]['interest_level'].mean()
                    st.info(f"**Premium Segment:**\n\n$100K+ Income\n\nAvg Interest: {high_income_interest:.2f}/5")
                    
                    st.warning("**Recommended Action:**\n\nFocus premium campaigns on high-income segments")
                else:
                    st.info("Age and interest data needed for insights")

            # CHART 2: Revenue Potential Heatmap
            st.markdown("---")
            st.subheader("2️⃣ Revenue Potential Heatmap: Monthly Revenue per Customer")
            st.markdown("**Action:** Prioritize segments with highest revenue potential")
            
            required_cols = ['age_group', 'income', 'monthly_revenue']
            if all(safe_column_check(df, col) for col in required_cols):
                revenue_heatmap = df.pivot_table(
                    values='monthly_revenue',
                    index='age_group',
                    columns='income',
                    aggfunc='mean'
                )
                
                column_order = ['Less than $25,000', '$25,000 - $49,999', '$50,000 - $74,999',
                                '$75,000 - $99,999', '$100,000 - $149,999', '$150,000 - $199,999', '$200,000+']
                existing_columns = [col for col in column_order if col in revenue_heatmap.columns]
                if existing_columns:
                    revenue_heatmap = revenue_heatmap[existing_columns]
                
                fig2 = go.Figure(data=go.Heatmap(
                    z=revenue_heatmap.values,
                    x=revenue_heatmap.columns,
                    y=revenue_heatmap.index,
                    colorscale='RdYlGn',
                    text=revenue_heatmap.values.round(2),
                    texttemplate='$%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Monthly<br>Revenue ($)")
                ))
                
                fig2.update_layout(
                    title='Average Monthly Revenue Potential per Customer by Age & Income',
                    xaxis_title='Annual Household Income',
                    yaxis_title='Age Group',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_revenue = df['monthly_revenue'].max()
                    st.metric("Max Monthly Revenue", f"${max_revenue:.2f}", "per customer")
                with col2:
                    avg_revenue = df[df['interest_level'] >= 4]['monthly_revenue'].mean()
                    st.metric("Avg Revenue (High Interest)", f"${avg_revenue:.2f}")
                with col3:
                    total_potential = df[df['interest_level'] >= 4]['monthly_revenue'].sum()
                    st.metric("Total Monthly Potential", f"${total_potential:,.0f}")
            else:
                st.warning("⚠️ Required columns for revenue heatmap not found")

            # CHART 3: Conversion Funnel Analysis
            st.markdown("---")
            st.subheader("3️⃣ Customer Conversion Funnel: From Awareness to Purchase")
            st.markdown("**Action:** Optimize each stage to reduce drop-off")
            
            funnel_data = {}
            funnel_data['Total Respondents'] = len(df)
            
            barrier_cols = [col for col in df.columns if 'barrier' in col.lower() and 'boring' in col.lower()]
            if barrier_cols:
                funnel_data['Find Water Boring'] = df[barrier_cols[0]].sum()
            
            used_product_cols = [col for col in df.columns if col.startswith('used_product_')]
            if used_product_cols:
                funnel_data['Currently Use Enhancement'] = df[used_product_cols].sum(axis=1).gt(0).sum()
            
            if safe_column_check(df, 'interest_level'):
                funnel_data['Interested (≥3)'] = (df['interest_level'] >= 3).sum()
                funnel_data['High Interest (≥4)'] = (df['interest_level'] >= 4).sum()
            
            if safe_column_check(df, 'purchase_likelihood'):
                funnel_data['Likely to Buy'] = df['purchase_likelihood'].isin([
                    'Definitely would purchase', 'Probably would purchase'
                ]).sum()
            
            if safe_column_check(df, 'willingness_to_pay_continuous'):
                funnel_data['Premium Willing ($2+)'] = (df['willingness_to_pay_continuous'] >= 2.0).sum()
            
            if len(funnel_data) > 1:
                stages = list(funnel_data.keys())
                values = list(funnel_data.values())
                
                fig3 = go.Figure(go.Funnel(
                    y=stages,
                    x=values,
                    textposition="inside",
                    textinfo="value+percent initial",
                    marker=dict(color=px.colors.sequential.Blues_r[:len(stages)])
                ))
                
                fig3.update_layout(
                    title='Customer Conversion Funnel Analysis',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                if len(values) >= 4:
                    st.markdown("#### 📊 Conversion Rates Between Stages")
                    cols = st.columns(min(4, len(values)-1))
                    
                    for i in range(min(3, len(values)-1)):
                        if values[i] > 0:
                            conv_rate = (values[i+1] / values[i]) * 100
                            stage_from = stages[i][:15] + "..." if len(stages[i]) > 15 else stages[i]
                            stage_to = stages[i+1][:15] + "..." if len(stages[i+1]) > 15 else stages[i+1]
                            cols[i].metric(f"{stage_from} → {stage_to}", f"{conv_rate:.1f}%")
            else:
                st.warning("⚠️ Insufficient data for funnel analysis")

            # CHART 4: Price Sensitivity Analysis
            st.markdown("---")
            st.subheader("4️⃣ Price Sensitivity Analysis: Optimal Pricing Strategy")
            st.markdown("**Action:** Set pricing tiers based on customer willingness to pay")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if safe_column_check(df, 'interest_level') and safe_column_check(df, 'willingness_to_pay_continuous'):
                    fig4a = go.Figure()
                    
                    for interest in sorted(df['interest_level'].unique(), reverse=True):
                        subset = df[df['interest_level'] == interest]
                        if len(subset) > 0:
                            colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
                            color_idx = int(interest) - 1 if interest <= 5 else 4
                            
                            fig4a.add_trace(go.Box(
                                y=subset['willingness_to_pay_continuous'],
                                name=f'Interest {int(interest)}',
                                marker_color=colors[color_idx] if 0 <= color_idx < len(colors) else '#3498db'
                            ))
                    
                    fig4a.update_layout(
                        title='Willingness to Pay Distribution by Interest Level',
                        yaxis_title='Willingness to Pay ($)',
                        xaxis_title='Interest Level',
                        height=400,
                        template='plotly_white',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig4a, use_container_width=True)
                else:
                    st.warning("⚠️ Interest level or WTP data not available")
            
            with col2:
                price_cols = [col for col in df.columns if col.startswith('price_') and col.endswith('_perception')]
                
                if price_cols:
                    price_points = [15, 25, 35, 45, 60]
                    acceptance_data = []
                    
                    for price in price_points:
                        col_name = f'price_{price}_perception'
                        if safe_column_check(df, col_name):
                            acceptable = df[col_name].isin(['Bargain', 'Acceptable']).sum()
                            acceptance_data.append({
                                'Price': f'${price}',
                                'Acceptance Rate': (acceptable / len(df)) * 100
                            })
                    
                    if acceptance_data:
                        price_df = pd.DataFrame(acceptance_data)
                        
                        fig4b = go.Figure(go.Bar(
                            x=price_df['Price'],
                            y=price_df['Acceptance Rate'],
                            marker_color=px.colors.sequential.RdYlGn_r[:len(price_df)],
                            text=price_df['Acceptance Rate'].round(1),
                            texttemplate='%{text}%',
                            textposition='outside'
                        ))
                        
                        fig4b.update_layout(
                            title='Price Acceptance Rate by Package Price',
                            yaxis_title='Acceptance Rate (%)',
                            xaxis_title='Package Price',
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig4b, use_container_width=True)
                    else:
                        st.info("Price perception data not available")
                else:
                    st.info("Price perception columns not found")
            
            st.markdown("#### 💰 Recommended Pricing Strategy")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("**Entry Tier**\n\n$0.50-$0.75 per cap\n\n$15-20 for 30-pack\n\nTarget: Price-sensitive segment")
            with col2:
                st.info("**Standard Tier**\n\n$1.00-$1.50 per cap\n\n$25-35 for 30-pack\n\nTarget: Mainstream market")
            with col3:
                st.warning("**Premium Tier**\n\n$1.75-$2.50 per cap\n\n$45-60 for 30-pack\n\nTarget: Health-conscious, high-income")

            # CHART 5: Feature Importance
            st.markdown("---")
            st.subheader("5️⃣ Key Drivers of Customer Interest: What Matters Most")
            st.markdown("**Action:** Focus marketing messages on top appeal factors")
            
            appeal_cols = [col for col in df.columns if col.startswith('appealing_') and 'Nothing' not in col]
            
            if appeal_cols and safe_column_check(df, 'interest_level'):
                appeal_data = []
                
                for col in appeal_cols:
                    feature = col.replace('appealing_', '').replace('_', ' ').title()
                    count = df[col].sum()
                    if count > 0:
                        high_interest_rate = df[df[col] == 1]['interest_level'].mean() if (df[col] == 1).sum() > 0 else 0
                        appeal_data.append({
                            'Feature': feature,
                            'Count': int(count),
                            'Percentage': (count / len(df)) * 100,
                            'Avg Interest': high_interest_rate
                        })
                
                if appeal_data:
                    appeal_df = pd.DataFrame(appeal_data).sort_values('Percentage', ascending=True).tail(10)
                    
                    fig5 = go.Figure()
                    
                    fig5.add_trace(go.Bar(
                        y=appeal_df['Feature'],
                        x=appeal_df['Percentage'],
                        orientation='h',
                        marker=dict(
                            color=appeal_df['Avg Interest'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Avg<br>Interest")
                        ),
                        text=appeal_df['Percentage'].round(1),
                        texttemplate='%{text}%',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Mentioned by: %{x:.1f}%<br>Avg Interest: %{marker.color:.2f}<extra></extra>'
                    ))
                    
                    fig5.update_layout(
                        title='Top 10 Most Appealing Product Features',
                        xaxis_title='Percentage of Respondents',
                        yaxis_title='',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.info("No appealing features data found")
            else:
                st.warning("⚠️ Appealing features data not available")
            
            # Customer Concerns
            st.markdown("#### ⚠️ Customer Concerns to Address")
            
            concern_cols = [col for col in df.columns if col.startswith('concern_') and 'No concerns' not in col]
            
            if concern_cols:
                concern_data = []
                
                for col in concern_cols:
                    concern = col.replace('concern_', '').replace('_', ' ').title()
                    count = df[col].sum()
                    if count > 0:
                        concern_data.append({'Concern': concern, 'Count': int(count)})
                
                if concern_data:
                    concern_df = pd.DataFrame(concern_data).sort_values('Count', ascending=False).head(5)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig5b = go.Figure(go.Bar(
                            x=concern_df['Concern'],
                            y=concern_df['Count'],
                            marker_color='#e74c3c',
                            text=concern_df['Count'],
                            textposition='outside'
                        ))
                        
                        fig5b.update_layout(
                            title='Top 5 Customer Concerns',
                            xaxis_title='',
                            yaxis_title='Number of Respondents',
                            height=400,
                            template='plotly_white'
                        )
                        fig5b.update_xaxes(tickangle=-45)
                        
                        st.plotly_chart(fig5b, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Action Plan")
                        st.markdown("""
                        **1. Address Environmental Concerns:**
                        - Develop recyclable/biodegradable caps
                        - Partner with sustainability organizations
                        
                        **2. Price Value Communication:**
                        - Emphasize cost per use vs alternatives
                        - Show monthly savings
                        
                        **3. Product Quality Assurance:**
                        - Highlight natural ingredients
                        - Provide dissolution guarantee
                        
                        **4. Compatibility:**
                        - Design universal-fit caps
                        - Create compatibility guide
                        """)
                else:
                    st.info("No concern data available")
            else:
                st.info("Concern data not available in dataset")
                
        except Exception as e:
            st.error(f"❌ Error in Marketing Insights: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # ========================================================================
    # TAB 2: ML ALGORITHMS
    # ========================================================================
    with tab2:
        st.header("🤖 Machine Learning Algorithms for Predictive Analytics")
        
        ml_tab1, ml_tab2, ml_tab3 = st.tabs([
            "📊 Classification",
            "🎯 Clustering",
            "📈 Regression"
        ])
        
        # CLASSIFICATION
        with ml_tab1:
            st.subheader("Classification: Predicting Customer Interest Level")
            st.markdown("**Goal:** Predict whether a customer will have high interest (≥4) in the product")
            
            if st.button("🚀 Run Classification Analysis", key="classify"):
                with st.spinner("Training classification models..."):
                    try:
                        df_class = df.copy()
                        
                        if not safe_column_check(df_class, 'interest_level'):
                            st.error("❌ 'interest_level' column not found in dataset. Cannot perform classification.")
                        else:
                            df_class['high_interest'] = (df_class['interest_level'] >= 4).astype(int)
                            
                            # Build feature list - only columns that exist
                            potential_features = [
                                'health_consciousness', 
                                'hydration_importance',
                                'early_adopter_score', 
                                'premium_willingness_score',
                                'sustainability_importance', 
                                'income_numeric',
                                'spending_numeric', 
                                'usage_numeric'
                            ]
                            
                            feature_cols = get_available_columns(df_class, potential_features)
                            
                            # Add barrier columns that exist
                            barrier_cols = [col for col in df_class.columns if col.startswith('barrier_')]
                            feature_cols.extend(barrier_cols)
                            
                            st.info(f"✓ Using {len(feature_cols)} features for classification")
                            
                            if len(feature_cols) < 2:
                                st.error(f"❌ Need at least 2 features. Only found: {', '.join(feature_cols)}")
                            else:
                                X = df_class[feature_cols].fillna(0)
                                y = df_class['high_interest']
                                
                                if len(X) < 10:
                                    st.error("❌ Not enough data points (need at least 10 rows)")
                                elif y.nunique() < 2:
                                    st.error("❌ Target variable has only one class")
                                else:
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.3, random_state=42, stratify=y
                                        )
                                    except ValueError:
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.3, random_state=42
                                        )
                                    
                                    models = {
                                        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
                                    }
                                    
                                    results = {}
                                    
                                    for name, model in models.items():
                                        try:
                                            model.fit(X_train, y_train)
                                            y_pred = model.predict(X_test)
                                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                                            
                                            results[name] = {
                                                'model': model,
                                                'y_pred': y_pred,
                                                'y_pred_proba': y_pred_proba,
                                                'accuracy': accuracy_score(y_test, y_pred),
                                                'precision': precision_score(y_test, y_pred, zero_division=0),
                                                'recall': recall_score(y_test, y_pred, zero_division=0),
                                                'f1': f1_score(y_test, y_pred, zero_division=0),
                                                'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
                                            }
                                        except Exception as model_error:
                                            st.warning(f"⚠️ Could not train {name}: {str(model_error)}")
                                    
                                    if len(results) == 0:
                                        st.error("❌ No models could be trained")
                                    else:
                                        st.success("✅ Classification models trained successfully!")
                                        
                                        st.markdown("### 📊 Model Performance Comparison")
                                        
                                        metrics_df = pd.DataFrame({
                                            'Model': list(results.keys()),
                                            'Accuracy': [r['accuracy'] for r in results.values()],
                                            'Precision': [r['precision'] for r in results.values()],
                                            'Recall': [r['recall'] for r in results.values()],
                                            'F1-Score': [r['f1'] for r in results.values()],
                                            'AUC-ROC': [r['auc'] for r in results.values()]
                                        })
                                        
                                        st.dataframe(metrics_df.style.format({
                                            'Accuracy': '{:.4f}',
                                            'Precision': '{:.4f}',
                                            'Recall': '{:.4f}',
                                            'F1-Score': '{:.4f}',
                                            'AUC-ROC': '{:.4f}'
                                        }).background_gradient(cmap='RdYlGn'))
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
                                            cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
                                            
                                            fig_cm = go.Figure(data=go.Heatmap(
                                                z=cm,
                                                x=['Predicted Low', 'Predicted High'],
                                                y=['Actual Low', 'Actual High'],
                                                text=cm,
                                                texttemplate='%{text}',
                                                colorscale='Blues'
                                            ))
                                            
                                            fig_cm.update_layout(
                                                title=f'Confusion Matrix - {best_model_name}',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig_cm, use_container_width=True)
                                        
                                        with col2:
                                            fig_roc = go.Figure()
                                            
                                            for name, result in results.items():
                                                try:
                                                    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                                                    fig_roc.add_trace(go.Scatter(
                                                        x=fpr, y=tpr,
                                                        mode='lines',
                                                        name=f'{name} (AUC={result["auc"]:.3f})',
                                                        line=dict(width=3)
                                                    ))
                                                except:
                                                    pass
                                            
                                            fig_roc.add_trace(go.Scatter(
                                                x=[0, 1], y=[0, 1],
                                                mode='lines',
                                                name='Random',
                                                line=dict(dash='dash', color='gray')
                                            ))
                                            
                                            fig_roc.update_layout(
                                                title='ROC Curve',
                                                xaxis_title='False Positive Rate',
                                                yaxis_title='True Positive Rate',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig_roc, use_container_width=True)
                                        
                                        if 'Random Forest' in results:
                                            st.markdown("### 🎯 Feature Importance")
                                            
                                            rf_model = results['Random Forest']['model']
                                            feature_importance = pd.DataFrame({
                                                'Feature': feature_cols,
                                                'Importance': rf_model.feature_importances_
                                            }).sort_values('Importance', ascending=True).tail(10)
                                            
                                            fig_fi = go.Figure(go.Bar(
                                                y=feature_importance['Feature'],
                                                x=feature_importance['Importance'],
                                                orientation='h',
                                                marker_color='#3498db'
                                            ))
                                            
                                            fig_fi.update_layout(
                                                title='Top 10 Most Important Features',
                                                xaxis_title='Importance',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig_fi, use_container_width=True)
                                        
                                        st.markdown("### 💡 Business Insights")
                                        best_result = results[best_model_name]
                                        st.info(f"""
                                        **Best Model:** {best_model_name}
                                        
                                        **Performance:**
                                        - Accuracy: {best_result['accuracy']*100:.2f}%
                                        - Precision: {best_result['precision']*100:.2f}%
                                        - Recall: {best_result['recall']*100:.2f}%
                                        - AUC-ROC: {best_result['auc']:.3f}
                                        
                                        **Application:** Use to prioritize high-potential customers
                                        """)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # CLUSTERING
        with ml_tab2:
            st.subheader("Clustering: Customer Segmentation Analysis")
            st.markdown("**Goal:** Identify distinct customer segments for targeted marketing")
            
            n_clusters = st.slider("Select number of clusters:", 2, 8, 4, key="n_clusters")
            
            if st.button("🚀 Run Clustering Analysis", key="cluster"):
                with st.spinner("Performing clustering..."):
                    try:
                        df_cluster = df.copy()
                        
                        cluster_features = []
                        
                        # Encode categorical variables
                        if safe_column_check(df_cluster, 'age_group'):
                            le_age = LabelEncoder()
                            df_cluster['age_encoded'] = le_age.fit_transform(df_cluster['age_group'].astype(str))
                            cluster_features.append('age_encoded')
                        
                        if safe_column_check(df_cluster, 'income'):
                            le_income = LabelEncoder()
                            df_cluster['income_encoded'] = le_income.fit_transform(df_cluster['income'].astype(str))
                            cluster_features.append('income_encoded')
                        
                        if safe_column_check(df_cluster, 'exercise_frequency'):
                            le_exercise = LabelEncoder()
                            df_cluster['exercise_encoded'] = le_exercise.fit_transform(df_cluster['exercise_frequency'].astype(str))
                            cluster_features.append('exercise_encoded')
                        
                        if safe_column_check(df_cluster, 'daily_water_intake'):
                            le_water = LabelEncoder()
                            df_cluster['water_encoded'] = le_water.fit_transform(df_cluster['daily_water_intake'].astype(str))
                            cluster_features.append('water_encoded')
                        
                        # Add numeric features
                        numeric_features = ['health_consciousness', 'interest_level', 
                                          'willingness_to_pay_continuous', 'early_adopter_score',
                                          'premium_willingness_score']
                        
                        cluster_features.extend(get_available_columns(df_cluster, numeric_features))
                        
                        if len(cluster_features) < 2:
                            st.error(f"❌ Need at least 2 features. Found: {', '.join(cluster_features)}")
                        else:
                            X_cluster = df_cluster[cluster_features].fillna(0)
                            
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_cluster)
                            
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
                            
                            silhouette = silhouette_score(X_scaled, df_cluster['cluster'])
                            davies_bouldin = davies_bouldin_score(X_scaled, df_cluster['cluster'])
                            calinski = calinski_harabasz_score(X_scaled, df_cluster['cluster'])
                            
                            st.success("✅ Clustering completed!")
                            
                            st.markdown("### 📊 Clustering Quality Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Silhouette Score", f"{silhouette:.4f}", "Higher is better")
                            col2.metric("Davies-Bouldin", f"{davies_bouldin:.4f}", "Lower is better")
                            col3.metric("Calinski-Harabasz", f"{calinski:.2f}", "Higher is better")
                            
                            st.markdown("### 🎯 Segment Visualization")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if safe_column_check(df_cluster, 'income_numeric') and safe_column_check(df_cluster, 'interest_level'):
                                    fig_cluster1 = px.scatter(
                                        df_cluster,
                                        x='income_numeric',
                                        y='interest_level',
                                        color='cluster',
                                        title='Segments: Income vs Interest',
                                        color_continuous_scale='Viridis'
                                    )
                                    fig_cluster1.update_layout(height=400)
                                    st.plotly_chart(fig_cluster1, use_container_width=True)
                                else:
                                    st.info("Income or interest data not available")
                            
                            with col2:
                                cluster_sizes = df_cluster['cluster'].value_counts().sort_index()
                                
                                fig_cluster2 = go.Figure(data=[go.Pie(
                                    labels=[f'Segment {i}' for i in cluster_sizes.index],
                                    values=cluster_sizes.values,
                                    hole=0.4
                                )])
                                
                                fig_cluster2.update_layout(
                                    title='Segment Distribution',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_cluster2, use_container_width=True)
                            
                            st.markdown("### 📋 Segment Profiles")
                            
                            for cluster_id in range(n_clusters):
                                cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
                                
                                with st.expander(f"📊 Segment {cluster_id} ({len(cluster_data)} customers, {len(cluster_data)/len(df)*100:.1f}%)"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown("**Demographics**")
                                        if safe_column_check(cluster_data, 'age_group'):
                                            st.write(f"• Age: {cluster_data['age_group'].mode()[0] if len(cluster_data) > 0 else 'N/A'}")
                                        if safe_column_check(cluster_data, 'income'):
                                            st.write(f"• Income: {cluster_data['income'].mode()[0] if len(cluster_data) > 0 else 'N/A'}")
                                    
                                    with col2:
                                        st.markdown("**Behavior**")
                                        if safe_column_check(cluster_data, 'health_consciousness'):
                                            st.write(f"• Health: {cluster_data['health_consciousness'].mean():.2f}/5")
                                        if safe_column_check(cluster_data, 'interest_level'):
                                            st.write(f"• Interest: {cluster_data['interest_level'].mean():.2f}/5")
                                    
                                    with col3:
                                        st.markdown("**Commercial**")
                                        if safe_column_check(cluster_data, 'willingness_to_pay_continuous'):
                                            st.write(f"• Avg WTP: ${cluster_data['willingness_to_pay_continuous'].mean():.2f}")
                                        if safe_column_check(cluster_data, 'monthly_revenue'):
                                            st.write(f"• Revenue: ${cluster_data['monthly_revenue'].mean():.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # REGRESSION
        with ml_tab3:
            st.subheader("Regression: Predicting Willingness to Pay")
            st.markdown("**Goal:** Predict how much a customer is willing to pay per cap")
            
            if st.button("🚀 Run Regression Analysis", key="regress"):
                with st.spinner("Training regression models..."):
                    try:
                        df_reg = df.copy()
                        
                        if not safe_column_check(df_reg, 'willingness_to_pay_continuous'):
                            st.error("❌ 'willingness_to_pay_continuous' not found")
                        else:
                            potential_features = [
                                'interest_level', 'health_consciousness', 'income_numeric',
                                'early_adopter_score', 'premium_willingness_score',
                                'spending_numeric', 'hydration_importance'
                            ]
                            
                            feature_cols = get_available_columns(df_reg, potential_features)
                            
                            if safe_column_check(df_reg, 'exercise_frequency'):
                                le_exercise = LabelEncoder()
                                df_reg['exercise_encoded'] = le_exercise.fit_transform(df_reg['exercise_frequency'].astype(str))
                                feature_cols.append('exercise_encoded')
                            
                            if safe_column_check(df_reg, 'age_group'):
                                le_age = LabelEncoder()
                                df_reg['age_encoded'] = le_age.fit_transform(df_reg['age_group'].astype(str))
                                feature_cols.append('age_encoded')
                            
                            if len(feature_cols) < 2:
                                st.error(f"❌ Need at least 2 features. Found: {', '.join(feature_cols)}")
                            else:
                                X_reg = df_reg[feature_cols].fillna(0)
                                y_reg = df_reg['willingness_to_pay_continuous']
                                
                                valid_idx = ~y_reg.isna()
                                X_reg = X_reg[valid_idx]
                                y_reg = y_reg[valid_idx]
                                
                                if len(X_reg) < 10:
                                    st.error("❌ Not enough valid data points")
                                else:
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_reg, y_reg, test_size=0.3, random_state=42
                                    )
                                    
                                    models_reg = {
                                        'Linear Regression': LinearRegression(),
                                        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                                    }
                                    
                                    results_reg = {}
                                    
                                    for name, model in models_reg.items():
                                        model.fit(X_train, y_train)
                                        y_pred = model.predict(X_test)
                                        
                                        results_reg[name] = {
                                            'model': model,
                                            'y_pred': y_pred,
                                            'mse': mean_squared_error(y_test, y_pred),
                                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                            'mae': mean_absolute_error(y_test, y_pred),
                                            'r2': r2_score(y_test, y_pred)
                                        }
                                    
                                    st.success("✅ Regression models trained!")
                                    
                                    st.markdown("### 📊 Model Performance")
                                    
                                    metrics_reg_df = pd.DataFrame({
                                        'Model': list(results_reg.keys()),
                                        'R² Score': [r['r2'] for r in results_reg.values()],
                                        'RMSE': [r['rmse'] for r in results_reg.values()],
                                        'MAE': [r['mae'] for r in results_reg.values()]
                                    })
                                    
                                    st.dataframe(metrics_reg_df.style.format({
                                        'R² Score': '{:.4f}',
                                        'RMSE': '{:.4f}',
                                        'MAE': '{:.4f}'
                                    }).background_gradient(cmap='RdYlGn', subset=['R² Score']))
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        best_model_name = max(results_reg.keys(), key=lambda k: results_reg[k]['r2'])
                                        
                                        fig_pred = go.Figure()
                                        
                                        fig_pred.add_trace(go.Scatter(
                                            x=y_test,
                                            y=results_reg[best_model_name]['y_pred'],
                                            mode='markers',
                                            marker=dict(color='#3498db', size=8, opacity=0.6)
                                        ))
                                        
                                        fig_pred.add_trace(go.Scatter(
                                            x=[y_test.min(), y_test.max()],
                                            y=[y_test.min(), y_test.max()],
                                            mode='lines',
                                            line=dict(dash='dash', color='red')
                                        ))
                                        
                                        fig_pred.update_layout(
                                            title=f'Actual vs Predicted - {best_model_name}',
                                            xaxis_title='Actual WTP ($)',
                                            yaxis_title='Predicted WTP ($)',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_pred, use_container_width=True)
                                    
                                    with col2:
                                        residuals = y_test - results_reg[best_model_name]['y_pred']
                                        
                                        fig_resid = go.Figure()
                                        
                                        fig_resid.add_trace(go.Histogram(
                                            x=residuals,
                                            nbinsx=30,
                                            marker_color='#e74c3c'
                                        ))
                                        
                                        fig_resid.update_layout(
                                            title='Prediction Error Distribution',
                                            xaxis_title='Residual',
                                            yaxis_title='Frequency',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_resid, use_container_width=True)
                                    
                                    st.markdown("### 💡 Business Insights")
                                    
                                    r2 = results_reg[best_model_name]['r2']
                                    mae = results_reg[best_model_name]['mae']
                                    
                                    st.info(f"""
                                    **Best Model:** {best_model_name}
                                    
                                    **Performance:**
                                    - R² Score: {r2:.4f} (explains {r2*100:.1f}% of variance)
                                    - MAE: ${mae:.2f} (average error)
                                    - RMSE: ${results_reg[best_model_name]['rmse']:.2f}
                                    
                                    **Application:** Predict revenue and optimize pricing
                                    """)
                                    
                                    if best_model_name == 'Random Forest':
                                        st.markdown("### 🎯 Feature Importance")
                                        
                                        rf_model = results_reg['Random Forest']['model']
                                        feature_importance = pd.DataFrame({
                                            'Feature': feature_cols,
                                            'Importance': rf_model.feature_importances_
                                        }).sort_values('Importance', ascending=True).tail(10)
                                        
                                        fig_fi = go.Figure(go.Bar(
                                            y=feature_importance['Feature'],
                                            x=feature_importance['Importance'],
                                            orientation='h',
                                            marker_color='#9b59b6'
                                        ))
                                        
                                        fig_fi.update_layout(
                                            title='Top Features Influencing WTP',
                                            xaxis_title='Importance',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_fi, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # ========================================================================
    # TAB 3: PREDICTION TOOL
    # ========================================================================
    with tab3:
        st.header("🔮 Customer Interest Prediction Tool")
        st.markdown("Upload new customer data to predict their interest level and purchase likelihood")
        
        st.markdown("### 📤 Upload New Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with customer data",
            type=['csv'],
            help="Upload a CSV file with customer features"
        )
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded! {len(new_data)} records found.")
                
                with st.expander("📋 Preview Data (first 10 rows)"):
                    st.dataframe(new_data.head(10))
                
                required_cols = [
                    'health_consciousness', 'hydration_importance',
                    'early_adopter_score', 'premium_willingness_score',
                    'sustainability_importance'
                ]
                
                available_cols = get_available_columns(new_data, required_cols)
                
                if len(available_cols) < 3:
                    st.error(f"❌ Need at least 3 features. Found: {', '.join(available_cols)}")
                else:
                    if st.button("🚀 Generate Predictions"):
                        with st.spinner("Generating predictions..."):
                            try:
                                feature_cols = available_cols.copy()
                                
                                if 'income' in new_data.columns:
                                    new_data['income_numeric'] = new_data['income'].map(create_income_mapping()).fillna(3)
                                    feature_cols.append('income_numeric')
                                
                                if 'monthly_beverage_spend' in new_data.columns:
                                    new_data['spending_numeric'] = new_data['monthly_beverage_spend'].map(create_spending_mapping()).fillna(50)
                                    feature_cols.append('spending_numeric')
                                
                                barrier_cols = [col for col in new_data.columns if col.startswith('barrier_')]
                                feature_cols.extend(barrier_cols)
                                
                                X_new = new_data[feature_cols].fillna(0)
                                
                                # Train model
                                if safe_column_check(df, 'interest_level'):
                                    df_train = df.copy()
                                    df_train['high_interest'] = (df_train['interest_level'] >= 4).astype(int)
                                    
                                    X_train = df_train[feature_cols].fillna(0)
                                    y_train = df_train['high_interest']
                                    
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                    model.fit(X_train, y_train)
                                    
                                    predictions = model.predict(X_new)
                                    predictions_proba = model.predict_proba(X_new)[:, 1]
                                    
                                    new_data['predicted_interest_label'] = predictions
                                    new_data['predicted_interest_label'] = new_data['predicted_interest_label'].map({
                                        0: 'Low Interest (1-3)',
                                        1: 'High Interest (4-5)'
                                    })
                                    new_data['interest_probability'] = (predictions_proba * 100).round(2)
                                    
                                    new_data['predicted_purchase_likelihood'] = new_data['interest_probability'].apply(
                                        lambda x: 'Definitely would purchase' if x >= 80 else
                                                 'Probably would purchase' if x >= 60 else
                                                 'Might or might not purchase' if x >= 40 else
                                                 'Probably would not purchase' if x >= 20 else
                                                 'Definitely would not purchase'
                                    )
                                    
                                    new_data['predicted_wtp'] = (0.75 + (predictions_proba * 1.75)).round(2)
                                    
                                    st.success("✅ Predictions generated!")
                                    
                                    st.markdown("### 📊 Prediction Summary")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    high_interest_count = (predictions == 1).sum()
                                    avg_prob = predictions_proba.mean() * 100
                                    likely_buyers = new_data['predicted_purchase_likelihood'].isin([
                                        'Definitely would purchase', 'Probably would purchase'
                                    ]).sum()
                                    avg_wtp = new_data['predicted_wtp'].mean()
                                    
                                    col1.metric("High Interest", f"{high_interest_count}", 
                                               f"{high_interest_count/len(new_data)*100:.1f}%")
                                    col2.metric("Avg Probability", f"{avg_prob:.1f}%")
                                    col3.metric("Likely Buyers", f"{likely_buyers}", 
                                               f"{likely_buyers/len(new_data)*100:.1f}%")
                                    col4.metric("Avg WTP", f"${avg_wtp:.2f}")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        interest_dist = new_data['predicted_interest_label'].value_counts()
                                        
                                        fig_interest = go.Figure(data=[go.Pie(
                                            labels=interest_dist.index,
                                            values=interest_dist.values,
                                            hole=0.4,
                                            marker_colors=['#e74c3c', '#27ae60']
                                        )])
                                        
                                        fig_interest.update_layout(
                                            title='Predicted Interest Distribution',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_interest, use_container_width=True)
                                    
                                    with col2:
                                        purchase_dist = new_data['predicted_purchase_likelihood'].value_counts()
                                        
                                        fig_purchase = go.Figure(data=[go.Bar(
                                            x=purchase_dist.index,
                                            y=purchase_dist.values,
                                            marker_color=px.colors.sequential.RdYlGn_r
                                        )])
                                        
                                        fig_purchase.update_layout(
                                            title='Purchase Likelihood',
                                            xaxis_title='',
                                            yaxis_title='Count',
                                            height=400
                                        )
                                        fig_purchase.update_xaxes(tickangle=-45)
                                        
                                        st.plotly_chart(fig_purchase, use_container_width=True)
                                    
                                    st.markdown("### 📋 Detailed Predictions")
                                    
                                    display_cols = [
                                        'predicted_interest_label',
                                        'interest_probability',
                                        'predicted_purchase_likelihood',
                                        'predicted_wtp'
                                    ]
                                    
                                    if 'age_group' in new_data.columns:
                                        display_cols.insert(0, 'age_group')
                                    if 'income' in new_data.columns:
                                        display_cols.insert(1, 'income')
                                    
                                    st.dataframe(new_data[display_cols].head(20), use_container_width=True)
                                    
                                    st.markdown("### 💾 Download Predictions")
                                    
                                    csv = new_data.to_csv(index=False)
                                    
                                    st.download_button(
                                        label="📥 Download Predictions CSV",
                                        data=csv,
                                        file_name="customer_predictions.csv",
                                        mime="text/csv"
                                    )
                                    
                                    st.markdown("### 🎯 Marketing Recommendations")
                                    
                                    high_value = new_data[
                                        (new_data['predicted_interest_label'] == 'High Interest (4-5)') &
                                        (new_data['predicted_wtp'] >= 2.0)
                                    ]
                                    
                                    if len(high_value) > 0:
                                        st.success(f"""
                                        **High-Value Segment:** {len(high_value)} customers
                                        
                                        Action: Priority outreach with premium messaging
                                        """)
                                else:
                                    st.error("❌ Training data missing interest_level")
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file to start predictions")
            
            st.markdown("### 📝 Required Columns")
            st.markdown("""
            **Minimum (at least 3):**
            - `health_consciousness` (1-5)
            - `hydration_importance` (1-5)
            - `early_adopter_score` (1-5)
            - `premium_willingness_score` (1-5)
            - `sustainability_importance` (1-5)
            
            **Optional:**
            - `income`, `age_group`, `exercise_frequency`
            """)
            
            st.markdown("### 📥 Download Sample Template")
            
            sample_data = pd.DataFrame({
                'health_consciousness': [4, 3, 5, 2, 4],
                'hydration_importance': [4, 3, 5, 3, 4],
                'early_adopter_score': [3, 4, 5, 2, 3],
                'premium_willingness_score': [4, 3, 5, 2, 4],
                'sustainability_importance': [5, 4, 5, 3, 4],
                'income': ['$75,000 - $99,999', '$50,000 - $74,999', '$100,000 - $149,999', 
                          '$25,000 - $49,999', '$75,000 - $99,999'],
                'age_group': ['25-34', '35-44', '25-34', '45-54', '35-44']
            })
            
            sample_csv = sample_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sample CSV",
                data=sample_csv,
                file_name="sample_template.csv",
                mime="text/csv"
            )

    # ========================================================================
    # TAB 4: ASSOCIATION RULES
    # ========================================================================
    with tab4:
        st.header("🔗 Association Rules: Purchase Preference & Price Perception")
        st.markdown("Discover patterns between customer preferences and price perceptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider("Minimum Support (%)", 5, 50, 10, 5) / 100
        
        with col2:
            min_confidence = st.slider("Minimum Confidence (%)", 50, 95, 70, 5) / 100
        
        if st.button("🚀 Run Association Rule Mining"):
            with st.spinner("Mining association rules..."):
                try:
                    price_cols = [col for col in df.columns if col.startswith('price_') and col.endswith('_perception')]
                    
                    if not safe_column_check(df, 'purchase_preference'):
                        st.error("❌ 'purchase_preference' not found")
                    elif len(price_cols) == 0:
                        st.error("❌ No price perception columns found")
                    else:
                        transactions = []
                        
                        for idx, row in df.iterrows():
                            transaction = []
                            
                            if pd.notna(row['purchase_preference']):
                                transaction.append(f"PREF: {row['purchase_preference']}")
                            
                            for col in price_cols:
                                if pd.notna(row[col]):
                                    price = col.replace('price_', '').replace('_perception', '')
                                    perception = row[col]
                                    transaction.append(f"${price}: {perception}")
                            
                            if len(transaction) > 0:
                                transactions.append(transaction)
                        
                        if len(transactions) < 10:
                            st.error("❌ Not enough transaction data")
                        else:
                            te = TransactionEncoder()
                            te_ary = te.fit(transactions).transform(transactions)
                            transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
                            
                            frequent_itemsets = apriori(
                                transaction_df,
                                min_support=min_support,
                                use_colnames=True
                            )
                            
                            if len(frequent_itemsets) == 0:
                                st.warning("⚠️ No frequent itemsets found. Lower the support threshold.")
                            else:
                                rules = association_rules(
                                    frequent_itemsets,
                                    metric="confidence",
                                    min_threshold=min_confidence
                                )
                                
                                if len(rules) == 0:
                                    st.warning("⚠️ No rules found. Lower the confidence threshold.")
                                else:
                                    rules = rules.sort_values('lift', ascending=False)
                                    
                                    st.success(f"✅ Found {len(rules)} association rules!")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    col1.metric("Total Rules", len(rules))
                                    col2.metric("Avg Confidence", f"{rules['confidence'].mean()*100:.1f}%")
                                    col3.metric("Avg Lift", f"{rules['lift'].mean():.2f}x")
                                    col4.metric("Lift>1", f"{(rules['lift'] > 1).sum()}")
                                    
                                    st.markdown("### 🏆 Top 10 Rules")
                                    
                                    top_rules = rules.head(10)
                                    
                                    for i, (idx, row) in enumerate(top_rules.iterrows(), 1):
                                        ant_str = ', '.join(list(row['antecedents'])[:2])
                                        cons_str = ', '.join(list(row['consequents'])[:2])
                                        
                                        with st.expander(f"Rule {i}: {ant_str}... → {cons_str}..."):
                                            col1, col2 = st.columns([2, 1])
                                            
                                            with col1:
                                                st.markdown("**IF:**")
                                                for ant in list(row['antecedents'])[:5]:
                                                    st.write(f"  • {ant}")
                                                
                                                st.markdown("**THEN:**")
                                                for cons in list(row['consequents'])[:5]:
                                                    st.write(f"  • {cons}")
                                            
                                            with col2:
                                                st.metric("Support", f"{row['support']*100:.2f}%")
                                                st.metric("Confidence", f"{row['confidence']*100:.2f}%")
                                                st.metric("Lift", f"{row['lift']:.2f}x")
                                            
                                            st.info(f"""
                                            When customers have IF conditions, there's a 
                                            {row['confidence']*100:.1f}% chance of THEN conditions. 
                                            This is {row['lift']:.2f}x more likely than random.
                                            """)
                                    
                                    st.markdown("### 📊 Visualization")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig_assoc1 = go.Figure()
                                        
                                        fig_assoc1.add_trace(go.Scatter(
                                            x=top_rules['support'],
                                            y=top_rules['confidence'],
                                            mode='markers',
                                            marker=dict(
                                                size=top_rules['lift']*10,
                                                color=top_rules['lift'],
                                                colorscale='Viridis',
                                                showscale=True
                                            ),
                                            text=[f"Rule {i}" for i in range(1, len(top_rules)+1)]
                                        ))
                                        
                                        fig_assoc1.update_layout(
                                            title='Support vs Confidence',
                                            xaxis_title='Support',
                                            yaxis_title='Confidence',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_assoc1, use_container_width=True)
                                    
                                    with col2:
                                        fig_assoc2 = go.Figure(go.Bar(
                                            y=[f"Rule {i}" for i in range(1, len(top_rules)+1)],
                                            x=top_rules['lift'].values,
                                            orientation='h',
                                            marker_color=top_rules['lift'].values,
                                            marker_colorscale='RdYlGn'
                                        ))
                                        
                                        fig_assoc2.update_layout(
                                            title='Top Rules by Lift',
                                            xaxis_title='Lift',
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_assoc2, use_container_width=True)
                                    
                                    st.markdown("### 💾 Download Rules")
                                    
                                    export_df = top_rules.copy()
                                    export_df['antecedents'] = export_df['antecedents'].apply(lambda x: ', '.join(list(x)))
                                    export_df['consequents'] = export_df['consequents'].apply(lambda x: ', '.join(list(x)))
                                    
                                    csv_rules = export_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv(index=False)
                                    
                                    st.download_button(
                                        label="📥 Download Association Rules CSV",
                                        data=csv_rules,
                                        file_name="association_rules.csv",
                                        mime="text/csv"
                                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    st.error("❌ Unable to load data")
    st.info("""
    **To fix:**
    1. Upload 'water_enhancement_survey_data.csv'
    2. Place in same directory as app.py
    3. Refresh the app
    """)
