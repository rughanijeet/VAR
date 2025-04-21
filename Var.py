# Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pydantic import BaseModel, Field, confloat
from scipy.stats import norm
from typing import List, Optional
from quantmod.timeseries import dailyReturn

# VaR Input object
class VaRInputs(BaseModel):
    confidence_level: float = Field(..., ge=0.90, le=1.0, description="The confidence level for the VaR calculation")
    lookback_period: int = Field(..., ge=1, description="Number of historical days for risk estimation")
    num_simulations: int = Field(10000, ge=1000, le=100000, description="Number of Monte Carlo simulations")
    portfolio_weights: Optional[List[confloat(ge=0, le=1)]] = Field(None, description="Weights of assets in the portfolio (None for single stock)")
    portfolio_returns: pd.DataFrame = Field(..., description="Historical returns of portfolio assets or single stock")
    is_single_stock: bool = Field(False, description="Flag to indicate single stock calculation")

    # allow Pydantic to accept arbitrary Python types that are not part of the standard Pydantic types.
    class Config:
        arbitrary_types_allowed = True
    
    
# Risk Metrics
class RiskMetrics:
    def __init__(self, inputs: VaRInputs):
        self.confidence_level = inputs.confidence_level
        self.lookback_period = inputs.lookback_period
        self.num_simulations = inputs.num_simulations
        self.returns = inputs.portfolio_returns
        self.is_single_stock = inputs.is_single_stock

        if self.is_single_stock:
            self.weights = np.array([1.0])  # Single stock, full weight
        else:
            if inputs.portfolio_weights is None:
                raise ValueError(
                    "Portfolio weights must be provided for portfolio VaR calculation"
                )
            self.weights = np.array(inputs.portfolio_weights)
            if len(self.weights) != self.returns.shape[1]:
                raise ValueError("Portfolio weights must match the number of assets")

    def parametric_var(self) -> float:
        mean_returns = np.mean(self.returns, axis=0)
        std = np.std(self.returns, axis=0)
        return self.weights @ norm.ppf(
            1 - self.confidence_level, loc=mean_returns, scale=std
        )

    def historical_var(self) -> float:
        portfolio_returns = (
            self.returns if self.is_single_stock else self.returns @ self.weights
        )
        return np.percentile(portfolio_returns, 100 * (1 - self.confidence_level))

    def monte_carlo_var(self) -> float:
        mean_returns = np.mean(self.returns, axis=0)
        cov_matrix = (
            np.cov(self.returns.T)
            if not self.is_single_stock
            else np.var(self.returns, axis=0)
        )

        simulated_returns = (
            np.random.normal(mean_returns, np.sqrt(cov_matrix), self.num_simulations)
            if self.is_single_stock
            else np.random.multivariate_normal(
                mean_returns, cov_matrix, self.num_simulations
            )
        )

        portfolio_simulated_returns = (
            simulated_returns
            if self.is_single_stock
            else simulated_returns @ self.weights
        )
        return np.percentile(
            portfolio_simulated_returns, 100 * (1 - self.confidence_level)
        )

    def expected_shortfall(self) -> float:
        portfolio_returns = (
            self.returns if self.is_single_stock else self.returns @ self.weights
        )
        var = self.historical_var()
        return np.mean(portfolio_returns[portfolio_returns <= var])


if __name__ == "__main__":
    engine = create_engine("sqlite:///../Nifty50")
    assets = sorted(["ICICIBANK", "ITC", "RELIANCE", "TCS", "ADANIENT"])
    single_stock = "ICICIBANK"

    # Query close price from database
    df = pd.DataFrame()
    for asset in assets:
        query = f"SELECT Date, Close FROM {asset}"
        with engine.connect() as connection:
            df1 = pd.read_sql_query(query, connection, index_col="Date")
            df1.columns = [asset]
        df = pd.concat([df, df1], axis=1)

    logreturn = dailyReturn(df).dropna()
    single_stock_returns = logreturn[[single_stock]].dropna()

    # Portfolio 
    portfolio_risk_metrics = RiskMetrics(
        VaRInputs(
            confidence_level=0.95,
            lookback_period=252,
            num_simulations=10000,
            portfolio_weights=[0.1212, 0.1056, 0.2724, 0.1506, 0.3503],
            portfolio_returns=logreturn,
            is_single_stock=False,
        )
    )

    # Single Stock 
    single_stock_risk_metrics = RiskMetrics(
            VaRInputs(
                confidence_level=0.95,
                lookback_period=252,
                num_simulations=10000,
                portfolio_weights=None,
                portfolio_returns=single_stock_returns,
                is_single_stock=True,
            )
        )   

    print("Portfolio VaR and CVaR for the given confidence level")
    print(f"Parametric VaR : {portfolio_risk_metrics.parametric_var():.4f}")
    print(f"Historical VaR : {portfolio_risk_metrics.historical_var():.4f}")
    print(f"Monte Carlo VaR : {portfolio_risk_metrics.monte_carlo_var():.4f}")
    print(f"Expected Shortfall : {portfolio_risk_metrics.expected_shortfall():.4f}")

    print(f"\nSingle Stock ({single_stock}) VaR and CVaR for the given confidence level")
    print(f"Parametric VaR : {single_stock_risk_metrics.parametric_var():.4f}")
    print(f"Historical VaR : {single_stock_risk_metrics.historical_var():.4f}")
    print(f"Monte Carlo VaR : {single_stock_risk_metrics.monte_carlo_var():.4f}")
    print(f"Expected Shortfall : {single_stock_risk_metrics.expected_shortfall():.4f}")
# Import libraries for visualization
import plotly.graph_objects as go
# Create the histogram of daily returns
fig = go.Figure()

# Add traces to the histogram
for stock in df.columns:
    fig.add_trace(go.Histogram(
            x=logreturn[stock] * 100,  
            # nbinsx=200, 
            name = stock,
            histnorm='probability density'
    ))

# Update the layout to add title and axis labels
fig.update_layout(
    title='Histogram of Daily Returns',
    xaxis=dict(title='Daily Return (%)', range=[-10, 10]),   
    yaxis_title='Probability Density',                
    width=1000,                               
    height=600, 
    showlegend=True                                
)

# Show the plot
fig.show()

# VaR Scaling
forecast_days = 5
svar = portfolio_risk_metrics.parametric_var()*np.sqrt(forecast_days)
print(f"Scaled VaR: {svar:.4f}")
