CREATE TABLE IF NOT EXISTS health_risk_data (
    City VARCHAR(50),
    Date DATE,
    AQI INTEGER,
    PM2_5 FLOAT,
    PM10 FLOAT,
    NO2 FLOAT,
    SO2 FLOAT,
    Temperature FLOAT,
    Humidity FLOAT,
    Health_Impact_Score FLOAT, -- Target Variable for regression
    Risk_Category VARCHAR(20)  -- Target Variable for classification
);