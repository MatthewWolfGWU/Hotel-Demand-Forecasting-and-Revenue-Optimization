# Hotel Demand Forecasting and Revenue Optimization

A comprehensive pricing and revenue management project predicting hotel booking demand using machine learning, time series analysis, and large language models (LLMs) for optimal pricing strategies in the hospitality industry.

## Overview

This project addresses the critical challenge of revenue optimization in the hotel industry by developing predictive models for daily booking demand. Using a publicly available hotel booking dataset with 119,000 individual bookings, the analysis aggregates data to the daily level and applies multiple modeling techniques to forecast demand as a function of price (Average Daily Rate), seasonality, hotel characteristics, and guest behavior. The project uniquely combines traditional machine learning with In-Context Learning (ICL) using large language models to derive optimal pricing strategies.

## Business Problem

**Challenge**: Hotels must optimize pricing and capacity decisions in a perishable inventory environment
**Impact**: Unsold rooms represent permanent revenue loss; poor forecasting leads to suboptimal pricing and staffing
**Solution**: Data-driven demand forecasting models enabling dynamic pricing, resource allocation, and revenue maximization

## Research Questions

1. **What variables carry the most strength in hotel booking demand?**
2. **What is the optimal pricing strategy for hotels in this industry?**
3. **How can In-Context Learning with LLMs optimize pricing for real-world applications?**

## Dataset

**Source**: Publicly available hotel booking dataset (Antonio et al., 2019)
**Original Data**: 119,000 individual bookings from 2015-2017
**Aggregated Data**: 731 daily observations after time-series transformation
**Geographic Scope**: International hotels (anonymized locations)

**Key Features**:
- **Target Variable**: Daily bookings (demand)
- **Pricing**: Average Daily Rate (ADR)
- **Seasonality**: Month, day of week, holiday indicators
- **Hotel Characteristics**: Hotel type (City vs. Resort), room assignments
- **Guest Behavior**: Lead time, customer type (Transient, Group, Contract), distribution channel
- **Booking Details**: Required parking spaces, special requests, adults, babies, waiting list days
- **Market Segments**: Direct, Corporate, Online TA, Offline TA, Groups, Complementary
- **Deposit Types**: No Deposit, Non-Refund, Refundable
- **Meal Plans**: BB (Bed & Breakfast), HB (Half Board)

**Holidays Included**: New Year's Day, Good Friday, Easter Sunday, Labor Day, Christmas Day

## Methodology

### Phase 1: Data Aggregation & Engineering

**Transformation Process**:
- Aggregated 119,000 booking-level records to 731 daily observations
- Filtered out canceled bookings to focus on realized demand
- Created time-series dataset with date as index

**Feature Engineering**:
- **Seasonality Indicators**: Month dummies, day-of-week dummies, holiday binary flag
- **Numerical Aggregations**: Mean ADR, mean lead time, mean adults, mean babies, mean special requests
- **Categorical Proportions**: Hotel type, room assignment, customer type, market segment, deposit type
- **Correlation Filtering**: Retained 28 numerical columns + seasonal indicators with |correlation| ≥ 0.15 with bookings

### Phase 2: Exploratory Data Analysis (EDA)

**Temporal Patterns**:
- Time series plot revealed no long-term trend, high day-to-day variability
- Maximum outlier: 448 bookings in a single day (mean: 150 bookings/day)

**Seasonal Analysis (Boxplots)**:

*By Month*:
- **ADR**: Peaks in July/August (summer), lows in November/December/January (fall/winter)
- **Bookings**: Slight increases spring to early summer, lows in late fall and December/January

*By Day of Week*:
- **ADR**: Relatively constant, slight increase on Fridays
- **Bookings**: Strong variation, highest on Thursday/Friday

*By Holiday*:
- No significant difference in ADR or bookings on holidays vs. non-holidays

**Distribution Analysis**:
- Both ADR and bookings exhibit approximately normal (Gaussian) distributions

### Phase 3: Machine Learning Models

All models trained on 80/20 train-test split. Evaluation metrics: **RMSE**, **MAPE**, **WMAPE**.

#### Model 1: Linear Regression

**Approach**: Ordinary Least Squares regression
**Seasonality Control**: Month + day-of-week + holiday dummies

**Performance**:
- Without controls: Overfitting (treats seasonality as true demand drivers)
- With confounder control: RMSE ~39
- With holiday control: **RMSE: 38.73**

#### Model 2: LASSO Regression ⭐ **BEST MODEL**

**Hyperparameters**:
- Alpha (L1 regularization): 0.1

**Performance**:
- **Training R²**: 0.707
- **Test R²**: 0.6456
- **Test RMSE**: 38.22 (best)
- **MAPE**: 0.183
- **WMAPE**: 0.186

**Key Advantages**:
- L1 regularization shrinks irrelevant coefficients to zero (automatic feature selection)
- 18 of 48 features shrunk to zero, 30 retained
- Reduces variance without significantly increasing bias

**Top Positive Predictors**:
1. **Transient Party customers**: +74.50 bookings
2. **Assigned Room A**: +71.38 bookings
3. **City Hotel**: +24.65 bookings
4. **Groups market segment**: +18.31 bookings
5. **October arrivals**: +15.38 bookings
6. **March arrivals**: +14.84 bookings
7. **May arrivals**: +13.29 bookings
8. **April arrivals**: +10.85 bookings
9. **Holiday**: +10.42 bookings
10. **Monday arrivals**: +5.16 bookings

**Top Negative Predictors**:
1. **Contract customers**: -128.56 bookings
2. **No Deposit**: -126.95 bookings
3. **BB meal plan**: -122.40 bookings
4. **Direct market**: -47.41 bookings
5. **August arrivals**: -24.81 bookings
6. **July arrivals**: -27.34 bookings
7. **December arrivals**: -3.32 bookings

#### Model 3: Random Forest

**Hyperparameters**:
- Number of trees: 100
- Sampling: Bootstrap
- Feature subsampling: sqrt

**Performance**:
- Without controls: RMSE 43.09
- With month/weekday controls: RMSE 41.02
- With all seasonal controls: **RMSE: 41.99**

**Strengths**: Captures non-linear relationships

#### Model 4: Gradient Boosting

**Hyperparameters**:
- Learning rate: 0.15
- Estimators: 2,000

**Performance**:
- Without controls: RMSE 40.66
- With seasonal controls: **RMSE: 39.06**

#### Model 5: ARIMA(1,1,1)

**Approach**: Time series model with autoregression, first-differencing, and moving average
**Performance**: **RMSE: 45.68**

**Interpretation**: Learns how previous day's change in bookings and forecast error predict next day's change

### Phase 4: In-Context Learning (ICL) with LLMs

**Methodology**:
- Binned ADR into meaningful price ranges
- Computed demand index (non-canceled bookings) per bin
- Provided price-demand pairs to LLM in prompt
- LLM inferred demand curve structure without formal parameter estimation

**Findings**:
- **Downward-sloping demand curve** confirmed
- High booking volume: $70-$110 ADR
- Moderate decline: $120-$150 ADR
- Steep drop: Beyond $150 ADR

**Comparison with LASSO**:
- **ICL**: Lightweight, interpretable, example-driven (price-only focus)
- **LASSO**: Comprehensive, multivariate, quantifies effect sizes with uncertainty
- **Consensus**: Both methods confirm strong negative price-demand relationship

## Model Comparison Summary

| Model | Test RMSE | Test MAPE | Test WMAPE | Notes |
|-------|-----------|-----------|------------|-------|
| **LASSO (α=0.1)** | **38.22** 🏆 | **0.183** | **0.186** | Best overall |
| Linear Regression | 38.73 | ~0.18 | ~0.18 | Strong with seasonality control |
| Gradient Boosting | 39.06 | ~0.18 | ~0.19 | Complex, slight improvement |
| Random Forest | 41.99 | ~0.19 | ~0.19 | Non-linear capture |
| ARIMA(1,1,1) | 45.68 | ~0.20 | ~0.20 | Time series approach |

**Key Insight**: Simple models (LASSO, Linear Regression) outperformed complex models (Gradient Boosting, Random Forest, ARIMA) due to effective regularization and seasonality control.

## Technologies Used

- **Python 3**: Primary programming language
- **Jupyter Notebook**: Interactive analysis environment (4 deliverables)
- **Libraries**:
  - `pandas`: Data manipulation and aggregation
  - `numpy`: Numerical computing
  - `matplotlib`: Visualization of seasonal trends and distributions
  - `scikit-learn`: Machine learning models and evaluation
    - LinearRegression, Lasso, RandomForestRegressor, GradientBoostingRegressor
    - train_test_split, cross_validation
    - mean_squared_error, mean_absolute_percentage_error
  - `statsmodels`: ARIMA time series modeling
  - Large Language Models: In-Context Learning for demand curve inference

## Key Findings

### Demand Drivers (From LASSO Feature Selection)

**Strongest Positive Impacts**:
1. Transient Party customers (+74.50 bookings)
2. City Hotels (+24.65 vs. Resort Hotels)
3. May arrivals (+13.29 vs. baseline)
4. Groups market segment (+18.31)

**Strongest Negative Impacts**:
1. Contract customers (-128.56 bookings)
2. BB meal plan (-122.40)
3. Direct market (-47.41)
4. Summer months (July: -27.34, August: -24.81)

### Seasonality Patterns

**Monthly**:
- **High Demand**: March, April, May, October
- **Low Demand**: July, August, December

**Day of Week**:
- **High Demand**: Thursday, Friday
- **Low Demand**: No specific low days (relatively stable)

**Holidays**: Minimal impact on demand

### Price Elasticity

- Strong negative relationship between ADR and bookings
- Optimal pricing range: $70-$110 (high volume)
- Demand drops significantly above $150 ADR

## Business Recommendations

### 1. Dynamic Pricing Strategy

**Surge Pricing Opportunities**:
- **May arrivals**: Increase ADR during this high-demand month (+13.29 bookings)
- **Thursday/Friday**: Implement weekend surge pricing
- **City Hotels**: Leverage higher demand in urban locations

**Discount Periods**:
- **December**: Reduce ADR to stimulate demand (-3.32 bookings baseline)
- **Late Fall**: Promotional pricing for low-demand periods

### 2. Staffing Management

**High Demand Forecasts**:
- Increase on-site staff during predicted high-volume days
- Optimize hourly wage costs based on occupancy projections

**Low Demand Forecasts**:
- Reduce staffing levels to control variable costs
- Cross-train employees for flexibility

### 3. Inventory Management

- Adjust cleaning supplies, food, and beverage orders based on forecasted occupancy
- Optimize procurement to match expected guest volume

### 4. Marketing & Customer Acquisition

**Target Segments**:
- **Transient Party customers**: Highest demand driver (+74.50), focus promotional efforts
- **Groups market segment**: Strong positive impact (+18.31)
- **City Hotels**: Concentrate marketing in urban markets

**Avoid Over-Reliance**:
- Contract customers show negative impact (-128.56) - likely reflects substitution effects

### 5. Channel Strategy

- **Direct bookings**: Show negative coefficient (-47.41) - may indicate pricing or convenience issues
- Optimize distribution channel mix based on coefficient insights

## Project Structure

```
.
├── 1) Data Aggregation/
│   ├── Data_Aggregation.ipynb          # Booking-level to daily aggregation
│   └── Data_Aggregation_Pdf.pdf        # Documentation
├── 2) EDA/
│   ├── EDA.ipynb                        # Exploratory data analysis
│   └── EDA_Pdf.pdf                      # EDA visualizations and insights
├── 3) Modeling/
│   ├── Modeling.ipynb                   # All ML models + ICL with LLM
│   └── Modeling_Pdf.pdf                 # Model results and comparison
├── 4) Project Report/
│   └── Hotel Demand Paper.pdf           # Comprehensive 13-page report
└── README.md                            # Project documentation
```

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Primary metric for model comparison
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error measure
- **WMAPE (Weighted Mean Absolute Percentage Error)**: Weighted version accounting for demand magnitude
- **R² (Coefficient of Determination)**: Variance explained by model

## Insights for Hotel Operations

1. **Revenue Optimization**: Use LASSO model coefficients to identify high-value customer segments and booking patterns
2. **Capacity Planning**: Forecast demand 80/20 split demonstrates robust out-of-sample prediction
3. **Competitive Positioning**: City hotels outperform resort hotels in bookings - focus investments accordingly
4. **Customer Lifetime Value**: Transient customers drive immediate volume; contract customers may require pricing adjustments

## Limitations

1. **Data Anonymization**: No hotel identity, geographic location, or customer demographics limits segment-specific modeling
2. **Temporal Scope**: Data from 2015-2017 may not reflect post-2019 travel behavior shifts (COVID-19 impact)
3. **Competitive Context**: No data on nearby hotels or competitor pricing restricts market-level forecasting
4. **Dynamic Pricing**: Aggregated daily data lacks granularity for intraday price adjustments
5. **Sample Size**: 731 daily observations may limit seasonal pattern learning

## Future Enhancements

### Data Improvements
1. **Recent Data**: Incorporate 2023-2025 bookings to reflect current travel patterns
2. **Geographic Details**: Include hotel location and local market characteristics
3. **Competitor Data**: Add nearby hotel pricing and occupancy for market-level modeling
4. **Customer Segmentation**: Integrate demographics, loyalty status, and booking history

### Modeling Advancements
1. **LSTM Neural Networks**: Capture long-term seasonal dependencies automatically without manual controls
2. **Ensemble Methods**: Combine LASSO, Gradient Boosting, and Random Forest predictions
3. **Real-Time Forecasting**: Deploy models for continuous demand prediction
4. **Multi-Hotel Models**: Hierarchical models accounting for property-specific effects

### Revenue Management Features
1. **Dynamic Pricing Engine**: Automated ADR adjustments based on real-time demand forecasts
2. **Overbooking Optimization**: Models for cancellation rates and optimal overbooking levels
3. **Length-of-Stay Pricing**: Multi-night stay discount strategies
4. **Event Integration**: Incorporate local events, conferences, and holidays for demand spikes

## Academic Context

**Course**: DNSC 4281 - Pricing and Revenue Management Analytics
**Institution**: George Washington University
**Date**: December 2025
**Deliverables**: 4-phase project (Data Aggregation, EDA, Modeling, Final Report)

## Contributors

- Matthew Wolf - mattwolf@gwu.edu
- Charlie Buckman - charlie.buckman@gwu.edu
- Kaylyn Phung - kaylyn.phung@gwu.edu
- Lauren Kim - lauren.kim1@gwu.edu

## References

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

## License

This project was completed as part of academic coursework at George Washington University. Dataset sourced from publicly available hotel booking data under academic use terms.

## Data Attribution

Hotel booking data originally published by Antonio et al. (2019) and made available for academic research purposes.
