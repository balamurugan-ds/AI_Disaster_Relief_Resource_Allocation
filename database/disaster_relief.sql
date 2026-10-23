CREATE DATABASE disaster_relief;

USE disaster_relief;


CREATE TABLE disasters (

    disaster_id INT AUTO_INCREMENT PRIMARY KEY,

    disaster_type VARCHAR(50),

    location VARCHAR(100),

    population_affected INT,

    disaster_severity INT,

    injured_people INT,

    displaced_people INT,

    duration_days INT,

    rainfall_mm FLOAT,

    infrastructure_damage FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE predictions (

    prediction_id INT AUTO_INCREMENT PRIMARY KEY,

    disaster_id INT,

    food_required INT,

    water_required INT,

    medicine_required INT,

    rescue_teams_required INT,

    shelter_required INT,

    priority_score FLOAT,

    priority_level VARCHAR(20),

    FOREIGN KEY (disaster_id)
    REFERENCES disasters(disaster_id)
);


CREATE TABLE resource_inventory (

    resource_id INT AUTO_INCREMENT PRIMARY KEY,

    location VARCHAR(100),

    food_available INT,

    water_available INT,

    medicine_available INT,

    rescue_teams_available INT,

    shelter_available INT
);