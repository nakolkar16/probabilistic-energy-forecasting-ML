# Probabilistic Energy Forecasting ML

## Overview

This project aims to build an end-to-end day-ahead probabilistic forecasting system for German residual load using public SMARD data.

Residual load is the portion of electricity demand that remains after accounting for wind and solar generation. It is a practically relevant forecasting target because it reflects the flexibility that still needs to be covered by dispatchable generation, storage, imports, or other balancing resources.

The goal is to predict the next 24 hours of hourly residual load and produce probabilistic forecasts in the form of:

- P10
- P50
- P90

This provides not only an expected forecast, but also an uncertainty range for each forecast hour.

## Project Objective

The objective of this project is to develop a reproducible machine learning pipeline that:

- ingests and preprocesses public SMARD time-series data
- constructs a canonical hourly dataset for residual load
- engineers temporal, lag-based, and rolling-window features
- trains baseline and machine learning forecasting models
- generates day-ahead probabilistic forecasts
- evaluates both forecast accuracy and uncertainty calibration
- supports lightweight next-day forecast generation

## Why This Project Matters

From a business and energy-systems perspective, residual load forecasting is useful because it helps estimate how much electricity demand still needs to be covered after variable renewable generation is considered.

This makes the output relevant for:

- flexibility planning
- balancing preparation
- market-oriented analysis
- more efficient integration of renewable energy into the grid

## Forecasting Target

The target variable in this project is **residual load**, defined conceptually as:

**Residual Load = Electricity Demand - Wind Generation - Solar Generation**

The forecast horizon is **day-ahead**, meaning the system predicts the next **24 hourly values**.