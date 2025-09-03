DROP TABLE customers;
DROP TABLE orders;

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
amount MONEY NOT NULL,
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
amount MONEY,
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