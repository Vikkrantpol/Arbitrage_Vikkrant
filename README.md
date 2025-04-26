# Arbitrage_Vikkrant
This repository contains arbitrage strategies for crypto 
![backtest_results](https://github.com/user-attachments/assets/bf4d9b1f-b554-4c12-8509-160551c0d1e2)
![price_differences](https://github.com/user-attachments/assets/3cfa01cc-5373-4469-a19e-631db03fbb26)


# Cryptocurrency Arbitrage Backtesting System

## Overview
This project implements a Python-based backtesting framework to evaluate arbitrage opportunities for Bitcoin (BTC) trading between Binance and Delta exchanges. The system processes high-frequency order book data, identifies profitable price differences, and simulates trades with real-world constraints like trading fees, slippage, and risk management. Built with pandas, NumPy, and matplotlib, it calculates performance metrics (e.g., Sharpe ratio, max drawdown) and generates visualizations to assess strategy effectiveness.

The project demonstrates my expertise in algorithmic trading, Python programming, and financial risk management, developed as part of my work as an independent retail trader and electrical engineering studies at IIT Goa.

## Features
- **Data Processing**: Aligns and cleans high-frequency BTC/USD data from Binance and Delta, handling over 10,000+ data points with 30-second timestamp tolerance.
- **Arbitrage Logic**: Evaluates two strategies (buy Binance/sell Delta, or vice versa), ensuring profits exceed a 0.05% threshold after fees and slippage.
- **Risk Management**: Limits risk to 1% of $15,000 initial capital per trade, with dynamic trade sizing capped at 0.05 BTC.
- **Performance Metrics**: Computes total profit, win rate, Sharpe ratio, max drawdown, and CAGR, with visualizations of capital growth and price differences.
- **Robustness**: Includes error handling, logging, and data validation to ensure reliable backtesting.

## Repository Structure

├── data/                    # Input data files (not included due to size)
│   ├── binance_btcusd.csv   # Sample Binance BTC/USD order book data
│   ├── delta_btcusd.csv     # Sample Delta BTC/USD order book data
├── logs/                    # Log files generated during backtesting
├── plots/                   # Output plots (e.g., capital growth, price differences)
├── results/                 # Output CSV files with trades and metrics!


