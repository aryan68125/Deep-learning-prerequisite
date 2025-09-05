DROP TABLE customers;
DROP TABLE orders;

DROP VIEW customer_orders_view;
DROP FUNCTION get_customer_orders();

-- Create a customers table
CREATE TABLE Customers (
id BIGSERIAL PRIMARY KEY,
name VARCHAR(255) NOT NULL,
phone_number VARCHAR(10) 
);

-- Insert data into the customers table
INSERT INTO Customers (name,phone_number) VALUES
('Rollex', '9972688590'),
('Ballistic', '1000688590'),
('Barricade', '2472688590'),
('Bullet', '7622688590'),
('Barbatos', '1172688590'),
('Artillery', '1502688590'),
('Shotgun', '1852688590'),
('Kombat', '3652688590'),
('Lockdown', '1442688590'),
('Ignition', '6000688590'),
('Shockwave', '3602688590'),
('PointBreak', '7472688590'),
('GroundZero', '8002688590'),
('Interceptor','9112688590');

-- check if the data has been successfully inserted into the customers table
SELECT * FROM Customers;

-- Create an orders table
CREATE TABLE Orders (
id BIGSERIAL PRIMARY KEY,
customer_id BIGINT NOT NULL,
name VARCHAR(255) NOT NULL,
amount NUMERIC(12,2) NOT NULL,
order_date TIMESTAMP DEFAULT NOW(),
CONSTRAINT fk_customer FOREIGN KEY (customer_id)
REFERENCES Customers (id)
ON DELETE CASCADE
);

-- insert the data into the orders table
INSERT INTO Orders (customer_id,name,amount) VALUES
(1,'50 calibure fully automatic Browning machine gun',9978.123),
(2,'50 calibure desert eagle handgun',1000.123),
(3,'500 S and W Magnum',2470.123),
(4,'Anzio 20mm anti material sniper rifle',7620.123),
(5,'Beretta m9 9mm',1170.123),
(6,'Milkor MGL (40mm Multiple Grenade Launcher)',1500.123),
(7,'Benelli M4 (M1014) 12 gauge shotgun shell',1850.123),
(8,'AK-47',3650.123),
(9,'M249 SAW Mk 48 Mod 1 7.62×51mm',3650.123),
(10,'Javelin rocket launcher',6000.123),
(11,'Milkor MGL (40mm Multiple Grenade Launcher)',3600.123),
(12,'FN SCAR-H (Special Operations Forces Combat Assault Rifle - Heavy)',7470.123),
(13,'M249 SAW Mk 48 Mod 1 7.62×51mm',8000.123),
(14,'Beretta m9 9mm',9110.123);

-- check if the data is present in the orders table
SELECT * FROM Orders;

-- create a database function
CREATE OR REPLACE FUNCTION get_customer_orders()
RETURNS TABLE (
customer_id BIGINT,
customer_name VARCHAR,
phone_number VARCHAR,
order_id BIGINT,
order_name VARCHAR,
amount NUMERIC(12,2),
order_date TIMESTAMP
)
LANGUAGE plpgsql
AS $$
BEGIN

	RETURN QUERY
	SELECT 
		c.id AS customer_id,
		c.name AS customer_name,
		c.phone_number,
		o.id AS order_id,
		o.name AS order_name,
		o.amount,
		o.order_date
	FROM customers c 
	INNER JOIN Orders o on c.id = o.customer_id ORDER BY o.order_date DESC;
END;
$$

-- create a database view from a FUNCTION
CREATE OR REPLACE VIEW customer_orders_view AS
SELECT * FROM get_customer_orders();

-- Fire up and see if the databaseview is working or NOT
SELECT * FROM customer_orders_view;








-- SQL code that helps simulate handling large dataset in pandas --
-- voters_trend.sql
-- PostgreSQL script to create voters_trend table and insert synthetic data

DROP TABLE IF EXISTS voters_trend;

CREATE TABLE voters_trend (
    voter_id BIGSERIAL PRIMARY KEY,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    party_voted VARCHAR(50),
    vote_date TIMESTAMP NOT NULL DEFAULT NOW(),
    turnout BOOLEAN NOT NULL
);

-- Insert synthetic data (sample of ~50 rows, expand later)
INSERT INTO voters_trend (state, district, age, gender, party_voted, vote_date, turnout) VALUES
('Uttar Pradesh', 'Lucknow', 34, 'Male', 'BJP', '2024-04-19 10:30:00', TRUE),
('Uttar Pradesh', 'Lucknow', 29, 'Female', 'INC', '2024-04-19 11:15:00', TRUE),
('Uttar Pradesh', 'Kanpur', 41, 'Male', 'BJP', '2024-04-19 12:45:00', TRUE),
('Uttar Pradesh', 'Kanpur', 55, 'Female', 'BSP', '2024-04-19 13:20:00', TRUE),
('Maharashtra', 'Mumbai', 26, 'Male', 'SS', '2024-04-20 09:10:00', TRUE),
('Maharashtra', 'Mumbai', 33, 'Female', 'INC', '2024-04-20 10:05:00', TRUE),
('Maharashtra', 'Pune', 38, 'Male', 'BJP', '2024-04-20 11:50:00', TRUE),
('Maharashtra', 'Pune', 22, 'Female', 'AAP', '2024-04-20 12:30:00', TRUE),
('Bihar', 'Patna', 44, 'Male', 'RJD', '2024-04-21 10:40:00', TRUE),
('Bihar', 'Patna', 36, 'Female', 'BJP', '2024-04-21 11:25:00', TRUE),
('Bihar', 'Gaya', 51, 'Male', 'JD(U)', '2024-04-21 12:10:00', TRUE),
('Bihar', 'Gaya', 27, 'Female', 'INC', '2024-04-21 13:00:00', TRUE),
('Delhi', 'New Delhi', 30, 'Male', 'AAP', '2024-04-22 09:50:00', TRUE),
('Delhi', 'New Delhi', 25, 'Female', 'BJP', '2024-04-22 10:35:00', TRUE),
('Delhi', 'South Delhi', 42, 'Male', 'INC', '2024-04-22 11:15:00', TRUE),
('Delhi', 'South Delhi', 37, 'Female', 'AAP', '2024-04-22 12:00:00', TRUE),
('West Bengal', 'Kolkata', 33, 'Male', 'TMC', '2024-04-23 09:25:00', TRUE),
('West Bengal', 'Kolkata', 29, 'Female', 'BJP', '2024-04-23 10:15:00', TRUE),
('West Bengal', 'Darjeeling', 48, 'Male', 'INC', '2024-04-23 11:05:00', TRUE),
('West Bengal', 'Darjeeling', 39, 'Female', 'BJP', '2024-04-23 12:20:00', TRUE),
('Tamil Nadu', 'Chennai', 31, 'Male', 'DMK', '2024-04-24 09:40:00', TRUE),
('Tamil Nadu', 'Chennai', 28, 'Female', 'AIADMK', '2024-04-24 10:25:00', TRUE),
('Tamil Nadu', 'Coimbatore', 52, 'Male', 'BJP', '2024-04-24 11:10:00', TRUE),
('Tamil Nadu', 'Coimbatore', 45, 'Female', 'DMK', '2024-04-24 12:05:00', TRUE),
('Karnataka', 'Bengaluru', 27, 'Male', 'BJP', '2024-04-25 09:55:00', TRUE),
('Karnataka', 'Bengaluru', 34, 'Female', 'INC', '2024-04-25 10:40:00', TRUE),
('Karnataka', 'Mysuru', 47, 'Male', 'JDS', '2024-04-25 11:30:00', TRUE),
('Karnataka', 'Mysuru', 29, 'Female', 'BJP', '2024-04-25 12:15:00', TRUE),
('Kerala', 'Thiruvananthapuram', 40, 'Male', 'CPI(M)', '2024-04-26 09:20:00', TRUE),
('Kerala', 'Thiruvananthapuram', 35, 'Female', 'INC', '2024-04-26 10:05:00', TRUE),
('Kerala', 'Kochi', 50, 'Male', 'INC', '2024-04-26 11:10:00', TRUE),
('Kerala', 'Kochi', 32, 'Female', 'CPI(M)', '2024-04-26 12:25:00', TRUE),
('Punjab', 'Amritsar', 28, 'Male', 'INC', '2024-04-27 09:35:00', TRUE),
('Punjab', 'Amritsar', 36, 'Female', 'BJP', '2024-04-27 10:45:00', TRUE),
('Punjab', 'Ludhiana', 44, 'Male', 'AAP', '2024-04-27 11:30:00', TRUE),
('Punjab', 'Ludhiana', 30, 'Female', 'INC', '2024-04-27 12:10:00', TRUE),
('Rajasthan', 'Jaipur', 39, 'Male', 'BJP', '2024-04-28 09:50:00', TRUE),
('Rajasthan', 'Jaipur', 26, 'Female', 'INC', '2024-04-28 10:35:00', TRUE),
('Rajasthan', 'Udaipur', 53, 'Male', 'BJP', '2024-04-28 11:25:00', TRUE),
('Rajasthan', 'Udaipur', 41, 'Female', 'INC', '2024-04-28 12:00:00', TRUE),
('Haryana', 'Gurgaon', 29, 'Male', 'BJP', '2024-04-29 09:15:00', TRUE),
('Haryana', 'Gurgaon', 34, 'Female', 'INC', '2024-04-29 10:10:00', TRUE),
('Haryana', 'Faridabad', 46, 'Male', 'BJP', '2024-04-29 11:20:00', TRUE),
('Haryana', 'Faridabad', 33, 'Female', 'AAP', '2024-04-29 12:05:00', TRUE),
('Gujarat', 'Ahmedabad', 27, 'Male', 'BJP', '2024-04-30 09:30:00', TRUE),
('Gujarat', 'Ahmedabad', 31, 'Female', 'INC', '2024-04-30 10:15:00', TRUE),
('Gujarat', 'Surat', 42, 'Male', 'BJP', '2024-04-30 11:00:00', TRUE),
('Gujarat', 'Surat', 37, 'Female', 'INC', '2024-04-30 11:45:00', TRUE);

-- check if the data exists 
SELECT *  FROM voters_trend;

DROP TABLE voters_trend;


-- Read database data in chunks of 500 rows and send it to pandas to be saved in excel file
CREATE OR REPLACE FUNCTION get_voters_trend_chunk(
    limit_count INT,
    offset_count INT
)
RETURNS TABLE (
    id BIGINT,
    state VARCHAR(100),
    district VARCHAR(100),
    age INT,
    gender VARCHAR(10),
    party_voted VARCHAR(50),
    vote_date TIMESTAMP,
    turnout BOOLEAN
) AS $$
BEGIN 
    RETURN QUERY
    SELECT id, state, district, age, gender, party_voted, vote_date, turnout
    FROM voters_trend
    ORDER BY id
    LIMIT limit_count OFFSET offset_count;
END;
$$ LANGUAGE plpgsql STABLE;

-- Drop the function get_voters_trend_chunk
DROP FUNCTION get_voters_trend_chunk;
