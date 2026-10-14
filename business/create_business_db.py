"""Create a deterministic six-table business SQLite database and question set."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "business" / "business.sqlite"
DEFAULT_QUESTIONS = ROOT / "business" / "business_questions.json"

DDL = """
PRAGMA foreign_keys=ON;
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    city TEXT NOT NULL,
    segment TEXT NOT NULL,
    credit_limit REAL NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    cost_price REAL NOT NULL,
    stock_qty INTEGER NOT NULL,
    supplier_name TEXT NOT NULL,
    active INTEGER NOT NULL
);
CREATE TABLE quotations (
    quote_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    quote_date TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL,
    subtotal REAL NOT NULL,
    discount_amount REAL NOT NULL,
    total_amount REAL NOT NULL,
    sales_rep TEXT NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quote_id INTEGER REFERENCES quotations(quote_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    subtotal REAL NOT NULL,
    tax_amount REAL NOT NULL,
    total_amount REAL NOT NULL,
    sales_rep TEXT NOT NULL
);
CREATE TABLE invoices (
    invoice_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL UNIQUE REFERENCES orders(order_id),
    invoice_date TEXT NOT NULL,
    due_date TEXT NOT NULL,
    status TEXT NOT NULL,
    total_amount REAL NOT NULL,
    paid_amount REAL NOT NULL
);
CREATE TABLE expenses (
    expense_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    expense_date TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    amount REAL NOT NULL,
    vendor_name TEXT NOT NULL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_product ON orders(product_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_expenses_date ON expenses(expense_date);
"""

CUSTOMERS = [
    (1, "Acme Manufacturing", "acme@example.com", "Ha Noi", "Enterprise", 100000.0, "2023-02-11", "active"),
    (2, "Binh Minh Retail", "binhminh@example.com", "Ho Chi Minh City", "SME", 45000.0, "2023-05-18", "active"),
    (3, "Green Foods", "greenfoods@example.com", "Da Nang", "SME", 30000.0, "2023-07-02", "active"),
    (4, "Nova Tech", "novatech@example.com", "Ha Noi", "Enterprise", 120000.0, "2022-12-20", "active"),
    (5, "Mekong Retail", "mekong@example.com", "Can Tho", "Retail", 25000.0, "2024-01-10", "active"),
    (6, "Atlas Services", "atlas@example.com", "Ho Chi Minh City", "Enterprise", 90000.0, "2022-09-14", "inactive"),
    (7, "Sunrise Hotel", "sunrise@example.com", "Hai Phong", "SME", 28000.0, "2024-03-21", "active"),
    (8, "Lotus Education", "lotus@example.com", "Da Nang", "Retail", 18000.0, "2024-04-15", "active"),
]

PRODUCTS = [
    (1, "Cloud CRM", "Software", 1200.0, 500.0, 40, "Northwind Digital", 1),
    (2, "ERP Basic", "Software", 2500.0, 1100.0, 25, "Northwind Digital", 1),
    (3, "Barcode Scanner", "Hardware", 180.0, 90.0, 120, "Viet Devices", 1),
    (4, "Receipt Printer", "Hardware", 320.0, 180.0, 80, "Viet Devices", 1),
    (5, "Office Chair", "Office", 150.0, 85.0, 60, "Comfort Works", 1),
    (6, "Standing Desk", "Office", 420.0, 250.0, 35, "Comfort Works", 1),
    (7, "Data Migration", "Service", 3500.0, 1900.0, 0, "Northwind Digital", 1),
    (8, "Legacy License", "Software", 800.0, 300.0, 0, "Old Systems", 0),
    (9, "POS Terminal", "Hardware", 950.0, 510.0, 18, "Viet Devices", 1),
    (10, "Training Package", "Service", 600.0, 200.0, 0, "People First", 1),
]

QUOTATIONS = [
    (1, 1, "2024-01-05", "2024-02-05", "accepted", 5000.0, 250.0, 4750.0, "Lan"),
    (2, 2, "2024-01-12", "2024-02-12", "accepted", 1800.0, 0.0, 1800.0, "Minh"),
    (3, 3, "2024-02-03", "2024-03-03", "expired", 3500.0, 350.0, 3150.0, "Lan"),
    (4, 4, "2024-02-18", "2024-03-18", "accepted", 7000.0, 700.0, 6300.0, "Quang"),
    (5, 5, "2024-03-01", "2024-04-01", "sent", 1260.0, 60.0, 1200.0, "Minh"),
    (6, 6, "2024-03-11", "2024-04-11", "rejected", 2500.0, 0.0, 2500.0, "Quang"),
    (7, 7, "2024-04-04", "2024-05-04", "accepted", 3500.0, 175.0, 3325.0, "Lan"),
    (8, 8, "2024-04-20", "2024-05-20", "accepted", 1200.0, 120.0, 1080.0, "Minh"),
    (9, 1, "2024-05-02", "2024-06-02", "sent", 4200.0, 0.0, 4200.0, "Lan"),
    (10, 4, "2024-05-15", "2024-06-15", "accepted", 9500.0, 950.0, 8550.0, "Quang"),
]

ORDERS = [
    (1, 1, 2, 1, "2024-01-15", "delivered", 2, 2500.0, 5000.0, 500.0, 5500.0, "Lan"),
    (2, 2, 3, 2, "2024-01-20", "delivered", 5, 180.0, 900.0, 90.0, 990.0, "Minh"),
    (3, 3, 7, 3, "2024-02-09", "cancelled", 1, 3500.0, 3500.0, 350.0, 3850.0, "Lan"),
    (4, 4, 7, 4, "2024-02-24", "delivered", 2, 3500.0, 7000.0, 700.0, 7700.0, "Quang"),
    (5, 5, 5, 5, "2024-03-05", "shipped", 4, 150.0, 600.0, 60.0, 660.0, "Minh"),
    (6, 6, 2, 6, "2024-03-18", "cancelled", 1, 2500.0, 2500.0, 250.0, 2750.0, "Quang"),
    (7, 7, 7, 7, "2024-04-12", "delivered", 1, 3500.0, 3500.0, 350.0, 3850.0, "Lan"),
    (8, 8, 1, 8, "2024-04-25", "processing", 1, 1200.0, 1200.0, 120.0, 1320.0, "Minh"),
    (9, 1, 6, 9, "2024-05-10", "delivered", 3, 420.0, 1260.0, 126.0, 1386.0, "Lan"),
    (10, 4, 9, 10, "2024-05-21", "shipped", 3, 950.0, 2850.0, 285.0, 3135.0, "Quang"),
    (11, 2, 4, None, "2024-06-03", "delivered", 2, 320.0, 640.0, 64.0, 704.0, "Minh"),
    (12, 3, 10, None, "2024-06-10", "processing", 3, 600.0, 1800.0, 180.0, 1980.0, "Lan"),
    (13, 4, 1, None, "2024-06-17", "delivered", 4, 1200.0, 4800.0, 480.0, 5280.0, "Quang"),
    (14, 5, 3, None, "2024-07-02", "returned", 2, 180.0, 360.0, 36.0, 396.0, "Minh"),
    (15, 7, 5, None, "2024-07-08", "delivered", 10, 150.0, 1500.0, 150.0, 1650.0, "Lan"),
    (16, 8, 10, None, "2024-07-15", "shipped", 2, 600.0, 1200.0, 120.0, 1320.0, "Minh"),
    (17, 1, 9, None, "2024-08-01", "delivered", 5, 950.0, 4750.0, 475.0, 5225.0, "Lan"),
    (18, 2, 1, None, "2024-08-09", "processing", 1, 1200.0, 1200.0, 120.0, 1320.0, "Minh"),
]

INVOICES = [
    (1, 1, "2024-01-16", "2024-02-16", "paid", 5500.0, 5500.0),
    (2, 2, "2024-01-21", "2024-02-21", "paid", 990.0, 990.0),
    (3, 4, "2024-02-25", "2024-03-25", "paid", 7700.0, 7000.0),
    (4, 5, "2024-03-06", "2024-04-06", "open", 660.0, 0.0),
    (5, 7, "2024-04-13", "2024-05-13", "paid", 3850.0, 3850.0),
    (6, 8, "2024-04-26", "2024-05-26", "overdue", 1320.0, 500.0),
    (7, 9, "2024-05-11", "2024-06-11", "paid", 1386.0, 1386.0),
    (8, 10, "2024-05-22", "2024-06-22", "open", 3135.0, 1000.0),
    (9, 11, "2024-06-04", "2024-07-04", "paid", 704.0, 704.0),
    (10, 13, "2024-06-18", "2024-07-18", "open", 5280.0, 0.0),
    (11, 14, "2024-07-03", "2024-08-03", "refunded", 396.0, 396.0),
    (12, 15, "2024-07-09", "2024-08-09", "paid", 1650.0, 1650.0),
    (13, 16, "2024-07-16", "2024-08-16", "open", 1320.0, 0.0),
    (14, 17, "2024-08-02", "2024-09-02", "paid", 5225.0, 5225.0),
]

EXPENSES = [
    (1, 1, "2024-01-16", "shipping", "Delivery for ERP order", "approved", 180.0, "Fast Cargo"),
    (2, 2, "2024-01-21", "shipping", "Scanner delivery", "approved", 75.0, "Fast Cargo"),
    (3, 4, "2024-02-26", "contractor", "Migration subcontractor", "approved", 2300.0, "Data Helpers"),
    (4, 5, "2024-03-07", "shipping", "Chair delivery", "pending", 90.0, "Fast Cargo"),
    (5, 7, "2024-04-14", "contractor", "Implementation support", "approved", 1600.0, "Data Helpers"),
    (6, 8, "2024-04-27", "shipping", "CRM delivery", "approved", 120.0, "Fast Cargo"),
    (7, 9, "2024-05-12", "shipping", "Desk delivery", "approved", 140.0, "Fast Cargo"),
    (8, 10, "2024-05-23", "shipping", "POS delivery", "pending", 220.0, "Fast Cargo"),
    (9, 11, "2024-06-05", "shipping", "Printer delivery", "approved", 90.0, "Fast Cargo"),
    (10, 13, "2024-06-19", "contractor", "CRM setup", "approved", 700.0, "People First"),
    (11, None, "2024-07-01", "office", "Quarterly office supplies", "approved", 450.0, "Office Hub"),
    (12, 17, "2024-08-03", "shipping", "Terminal delivery", "approved", 260.0, "Fast Cargo"),
]


# Each tuple is (id, bilingual question, language, SQL, difficulty, tags).
QUESTION_SPECS = [
    ("q001", "How many customers are in the database?", "EN", "SELECT COUNT(*) AS customer_count FROM customers", "easy", ["count"]),
    ("q002", "Có bao nhiêu khách hàng đang hoạt động?", "VI", "SELECT COUNT(*) AS customer_count FROM customers WHERE status = 'active'", "easy", ["count", "filter"]),
    ("q003", "List all customer names alphabetically.", "EN", "SELECT customer_name FROM customers ORDER BY customer_name", "easy", ["order"]),
    ("q004", "Liệt kê khách hàng ở Hà Nội.", "VI", "SELECT customer_name FROM customers WHERE city = 'Ha Noi' ORDER BY customer_id", "easy", ["filter"]),
    ("q005", "What is the largest customer credit limit?", "EN", "SELECT MAX(credit_limit) AS max_credit_limit FROM customers", "easy", ["aggregate"]),
    ("q006", "Hạn mức tín dụng trung bình của khách hàng là bao nhiêu?", "VI", "SELECT AVG(credit_limit) AS average_credit_limit FROM customers", "easy", ["aggregate"]),
    ("q007", "Which customers belong to the Enterprise segment?", "EN", "SELECT customer_name FROM customers WHERE segment = 'Enterprise' ORDER BY customer_id", "easy", ["filter"]),
    ("q008", "Khách hàng nào được tạo từ năm 2024 trở đi?", "VI", "SELECT customer_name FROM customers WHERE created_at >= '2024-01-01' ORDER BY created_at", "easy", ["date", "filter"]),
    ("q009", "How many products are currently active?", "EN", "SELECT COUNT(*) AS active_products FROM products WHERE active = 1", "easy", ["count", "filter"]),
    ("q010", "Có bao nhiêu sản phẩm thuộc nhóm Hardware?", "VI", "SELECT COUNT(*) AS hardware_products FROM products WHERE category = 'Hardware'", "easy", ["count", "filter"]),
    ("q011", "Show product names and prices from highest to lowest price.", "EN", "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC, product_id", "easy", ["order"]),
    ("q012", "Sản phẩm nào đang hết hàng?", "VI", "SELECT product_name FROM products WHERE stock_qty = 0 ORDER BY product_id", "easy", ["filter"]),
    ("q013", "What is the total value of all product stock at selling price?", "EN", "SELECT SUM(stock_qty * unit_price) AS stock_value FROM products", "medium", ["aggregate", "arithmetic"]),
    ("q014", "Giá bán cao nhất của sản phẩm đang hoạt động là bao nhiêu?", "VI", "SELECT MAX(unit_price) AS max_active_price FROM products WHERE active = 1", "easy", ["aggregate", "filter"]),
    ("q015", "How many quotations were accepted?", "EN", "SELECT COUNT(*) AS accepted_quotes FROM quotations WHERE status = 'accepted'", "easy", ["count", "filter"]),
    ("q016", "Tổng giá trị báo giá đã chấp nhận là bao nhiêu?", "VI", "SELECT SUM(total_amount) AS accepted_quote_value FROM quotations WHERE status = 'accepted'", "easy", ["aggregate", "filter"]),
    ("q017", "List quotation IDs that expired before April 2024.", "EN", "SELECT quote_id FROM quotations WHERE status = 'expired' AND valid_until < '2024-04-01' ORDER BY quote_id", "medium", ["date", "filter"]),
    ("q018", "Báo giá trung bình theo nhân viên kinh doanh là bao nhiêu?", "VI", "SELECT sales_rep, AVG(total_amount) AS average_quote FROM quotations GROUP BY sales_rep ORDER BY sales_rep", "medium", ["group", "aggregate"]),
    ("q019", "Which customer received the highest quotation?", "EN", "SELECT c.customer_name, q.total_amount FROM quotations q JOIN customers c ON c.customer_id = q.customer_id ORDER BY q.total_amount DESC, q.quote_id LIMIT 1", "medium", ["join", "order"]),
    ("q020", "Có bao nhiêu báo giá của khách hàng doanh nghiệp?", "VI", "SELECT COUNT(*) AS enterprise_quotes FROM quotations q JOIN customers c ON c.customer_id = q.customer_id WHERE c.segment = 'Enterprise'", "medium", ["join", "count"]),
    ("q021", "How many orders were delivered?", "EN", "SELECT COUNT(*) AS delivered_orders FROM orders WHERE status = 'delivered'", "easy", ["count", "filter"]),
    ("q022", "Tổng doanh thu từ các đơn hàng đã giao là bao nhiêu?", "VI", "SELECT SUM(total_amount) AS delivered_revenue FROM orders WHERE status = 'delivered'", "easy", ["aggregate", "filter"]),
    ("q023", "List orders placed in June 2024.", "EN", "SELECT order_id, order_date FROM orders WHERE order_date >= '2024-06-01' AND order_date < '2024-07-01' ORDER BY order_date", "medium", ["date", "filter"]),
    ("q024", "Đơn hàng nào có tổng tiền lớn hơn 5.000?", "VI", "SELECT order_id, total_amount FROM orders WHERE total_amount > 5000 ORDER BY total_amount DESC", "easy", ["filter", "order"]),
    ("q025", "What is the average order value by order status?", "EN", "SELECT status, AVG(total_amount) AS average_order_value FROM orders GROUP BY status ORDER BY status", "medium", ["group", "aggregate"]),
    ("q026", "Số lượng sản phẩm đã bán theo từng mã sản phẩm là bao nhiêu?", "VI", "SELECT product_id, SUM(quantity) AS units_sold FROM orders GROUP BY product_id ORDER BY product_id", "medium", ["group", "aggregate"]),
    ("q027", "Which product generated the highest delivered revenue?", "EN", "SELECT p.product_name, SUM(o.total_amount) AS revenue FROM orders o JOIN products p ON p.product_id = o.product_id WHERE o.status = 'delivered' GROUP BY p.product_id, p.product_name ORDER BY revenue DESC LIMIT 1", "hard", ["join", "group", "aggregate"]),
    ("q028", "Khách hàng nào có tổng giá trị đơn hàng cao nhất?", "VI", "SELECT c.customer_name, SUM(o.total_amount) AS total_spend FROM orders o JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY total_spend DESC LIMIT 1", "hard", ["join", "group", "aggregate"]),
    ("q029", "How many orders do not have a quotation?", "EN", "SELECT COUNT(*) AS orders_without_quote FROM orders WHERE quote_id IS NULL", "easy", ["null", "count"]),
    ("q030", "Đơn hàng nào đang ở trạng thái processing hoặc shipped?", "VI", "SELECT order_id, status FROM orders WHERE status IN ('processing', 'shipped') ORDER BY order_id", "easy", ["filter"]),
    ("q031", "What is the total quantity ordered by Enterprise customers?", "EN", "SELECT SUM(o.quantity) AS enterprise_quantity FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.segment = 'Enterprise'", "medium", ["join", "aggregate"]),
    ("q032", "Doanh thu giao hàng theo thành phố khách hàng là bao nhiêu?", "VI", "SELECT c.city, SUM(o.total_amount) AS delivered_revenue FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE o.status = 'delivered' GROUP BY c.city ORDER BY c.city", "hard", ["join", "group"]),
    ("q033", "Which customer has never placed an order?", "EN", "SELECT c.customer_name FROM customers c LEFT JOIN orders o ON o.customer_id = c.customer_id WHERE o.order_id IS NULL ORDER BY c.customer_id", "hard", ["left_join", "null"]),
    ("q034", "Khách hàng nào đã đặt ít nhất hai đơn hàng?", "VI", "SELECT c.customer_name, COUNT(o.order_id) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY c.customer_id, c.customer_name HAVING COUNT(o.order_id) >= 2 ORDER BY c.customer_id", "hard", ["join", "having"]),
    ("q035", "What is the gross margin on delivered orders?", "EN", "SELECT SUM(o.quantity * (o.unit_price - p.cost_price)) AS gross_margin FROM orders o JOIN products p ON p.product_id = o.product_id WHERE o.status = 'delivered'", "hard", ["join", "arithmetic"]),
    ("q036", "Tổng tiền thuế của các đơn hàng đã giao là bao nhiêu?", "VI", "SELECT SUM(tax_amount) AS delivered_tax FROM orders WHERE status = 'delivered'", "easy", ["aggregate", "filter"]),
    ("q037", "How many invoices are overdue or still open?", "EN", "SELECT COUNT(*) AS unpaid_invoices FROM invoices WHERE status IN ('overdue', 'open')", "easy", ["count", "filter"]),
    ("q038", "Tổng số tiền còn phải thu trên các hóa đơn là bao nhiêu?", "VI", "SELECT SUM(total_amount - paid_amount) AS outstanding_amount FROM invoices WHERE status IN ('open', 'overdue')", "medium", ["arithmetic", "filter"]),
    ("q039", "List invoice IDs with a remaining balance.", "EN", "SELECT invoice_id, total_amount - paid_amount AS balance FROM invoices WHERE total_amount > paid_amount ORDER BY invoice_id", "medium", ["arithmetic", "filter"]),
    ("q040", "Hóa đơn nào đã thanh toán đầy đủ?", "VI", "SELECT invoice_id FROM invoices WHERE paid_amount = total_amount ORDER BY invoice_id", "easy", ["filter"]),
    ("q041", "What is the average invoice amount by invoice status?", "EN", "SELECT status, AVG(total_amount) AS average_invoice FROM invoices GROUP BY status ORDER BY status", "medium", ["group", "aggregate"]),
    ("q042", "Tổng tiền hóa đơn theo tháng phát hành là bao nhiêu?", "VI", "SELECT substr(invoice_date, 1, 7) AS invoice_month, SUM(total_amount) AS monthly_total FROM invoices GROUP BY invoice_month ORDER BY invoice_month", "hard", ["date", "group"]),
    ("q043", "Which customers have overdue invoices?", "EN", "SELECT DISTINCT c.customer_name FROM invoices i JOIN orders o ON o.order_id = i.order_id JOIN customers c ON c.customer_id = o.customer_id WHERE i.status = 'overdue' ORDER BY c.customer_name", "hard", ["join", "distinct"]),
    ("q044", "Khách hàng nào có tổng số tiền hóa đơn lớn nhất?", "VI", "SELECT c.customer_name, SUM(i.total_amount) AS invoice_total FROM invoices i JOIN orders o ON o.order_id = i.order_id JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY invoice_total DESC LIMIT 1", "hard", ["join", "group"]),
    ("q045", "How many expenses are approved?", "EN", "SELECT COUNT(*) AS approved_expenses FROM expenses WHERE status = 'approved'", "easy", ["count", "filter"]),
    ("q046", "Tổng chi phí theo danh mục là bao nhiêu?", "VI", "SELECT category, SUM(amount) AS total_expense FROM expenses GROUP BY category ORDER BY category", "medium", ["group", "aggregate"]),
    ("q047", "Which vendor received the highest total expense?", "EN", "SELECT vendor_name, SUM(amount) AS total_expense FROM expenses GROUP BY vendor_name ORDER BY total_expense DESC LIMIT 1", "medium", ["group", "order"]),
    ("q048", "Có bao nhiêu chi phí chưa được duyệt?", "VI", "SELECT COUNT(*) AS pending_expenses FROM expenses WHERE status = 'pending'", "easy", ["count", "filter"]),
    ("q049", "What was the total shipping expense in the second quarter of 2024?", "EN", "SELECT SUM(amount) AS shipping_q2_expense FROM expenses WHERE category = 'shipping' AND expense_date >= '2024-04-01' AND expense_date < '2024-07-01'", "medium", ["date", "aggregate"]),
    ("q050", "Chi phí nào không gắn với đơn hàng?", "VI", "SELECT expense_id, amount FROM expenses WHERE order_id IS NULL ORDER BY expense_id", "easy", ["null", "filter"]),
    ("q051", "Which order has the largest related expense?", "EN", "SELECT o.order_id, e.amount FROM expenses e JOIN orders o ON o.order_id = e.order_id ORDER BY e.amount DESC, o.order_id LIMIT 1", "medium", ["join", "order"]),
    ("q052", "Tổng chi phí liên quan đến các đơn đã giao là bao nhiêu?", "VI", "SELECT SUM(e.amount) AS delivered_order_expense FROM expenses e JOIN orders o ON o.order_id = e.order_id WHERE o.status = 'delivered'", "hard", ["join", "aggregate"]),
    ("q053", "Which active product has the highest margin per unit?", "EN", "SELECT product_name, unit_price - cost_price AS margin FROM products WHERE active = 1 ORDER BY margin DESC LIMIT 1", "medium", ["arithmetic", "order"]),
    ("q054", "Sản phẩm nào có giá bán cao hơn giá trung bình?", "VI", "SELECT product_name, unit_price FROM products WHERE unit_price > (SELECT AVG(unit_price) FROM products) ORDER BY unit_price DESC", "hard", ["subquery", "aggregate"]),
    ("q055", "Which customers have a credit limit above the average?", "EN", "SELECT customer_name, credit_limit FROM customers WHERE credit_limit > (SELECT AVG(credit_limit) FROM customers) ORDER BY credit_limit DESC", "hard", ["subquery", "aggregate"]),
    ("q056", "Đơn hàng có giá trị cao hơn giá trị trung bình là những đơn nào?", "VI", "SELECT order_id, total_amount FROM orders WHERE total_amount > (SELECT AVG(total_amount) FROM orders) ORDER BY total_amount DESC", "hard", ["subquery", "aggregate"]),
    ("q057", "What percentage of customers are active?", "EN", "SELECT 100.0 * SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) / COUNT(*) AS active_percentage FROM customers", "hard", ["case", "aggregate"]),
    ("q058", "Tỷ lệ hóa đơn đã thanh toán đầy đủ là bao nhiêu phần trăm?", "VI", "SELECT 100.0 * SUM(CASE WHEN paid_amount = total_amount THEN 1 ELSE 0 END) / COUNT(*) AS paid_percentage FROM invoices", "hard", ["case", "aggregate"]),
    ("q059", "For each sales representative, show accepted quotation value and order value.", "EN", "SELECT s.sales_rep, COALESCE(q.quote_value, 0) AS accepted_quote_value, COALESCE(o.order_value, 0) AS order_value FROM (SELECT DISTINCT sales_rep FROM quotations UNION SELECT DISTINCT sales_rep FROM orders) s LEFT JOIN (SELECT sales_rep, SUM(total_amount) AS quote_value FROM quotations WHERE status = 'accepted' GROUP BY sales_rep) q ON q.sales_rep = s.sales_rep LEFT JOIN (SELECT sales_rep, SUM(total_amount) AS order_value FROM orders GROUP BY sales_rep) o ON o.sales_rep = s.sales_rep ORDER BY s.sales_rep", "hard", ["subquery", "join"]),
    ("q060", "Nhân viên nào có giá trị đơn hàng đã giao cao nhất?", "VI", "SELECT sales_rep, SUM(total_amount) AS delivered_value FROM orders WHERE status = 'delivered' GROUP BY sales_rep ORDER BY delivered_value DESC LIMIT 1", "medium", ["group", "order"]),
    ("q061", "Which customer placed the most units across all orders?", "EN", "SELECT c.customer_name, SUM(o.quantity) AS units FROM orders o JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY units DESC LIMIT 1", "hard", ["join", "group"]),
    ("q062", "Khách hàng nào có đơn hàng bị hủy?", "VI", "SELECT DISTINCT c.customer_name FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE o.status = 'cancelled' ORDER BY c.customer_name", "medium", ["join", "distinct"]),
    ("q063", "What is the most expensive product in each category?", "EN", "SELECT p.category, p.product_name, p.unit_price FROM products p WHERE p.unit_price = (SELECT MAX(p2.unit_price) FROM products p2 WHERE p2.category = p.category) ORDER BY p.category", "hard", ["correlated_subquery", "group"]),
    ("q064", "Trong mỗi trạng thái đơn hàng, đơn nào có tổng tiền lớn nhất?", "VI", "SELECT o.status, o.order_id, o.total_amount FROM orders o WHERE o.total_amount = (SELECT MAX(o2.total_amount) FROM orders o2 WHERE o2.status = o.status) ORDER BY o.status", "hard", ["correlated_subquery", "group"]),
    ("q065", "Which products have never appeared in an order?", "EN", "SELECT p.product_name FROM products p LEFT JOIN orders o ON o.product_id = p.product_id WHERE o.order_id IS NULL ORDER BY p.product_id", "hard", ["left_join", "null"]),
    ("q066", "Sản phẩm nào đã xuất hiện trong cả đơn hàng đã giao và đơn đang xử lý?", "VI", "SELECT p.product_name FROM products p JOIN orders delivered ON delivered.product_id = p.product_id AND delivered.status = 'delivered' JOIN orders processing ON processing.product_id = p.product_id AND processing.status = 'processing' GROUP BY p.product_id, p.product_name ORDER BY p.product_id", "hard", ["self_join", "group"]),
    ("q067", "How much revenue remains after approved expenses for delivered orders?", "EN", "SELECT (SELECT SUM(total_amount) FROM orders WHERE status = 'delivered') - (SELECT SUM(e.amount) FROM expenses e JOIN orders o ON o.order_id = e.order_id WHERE o.status = 'delivered' AND e.status = 'approved') AS net_revenue", "hard", ["subquery", "arithmetic"]),
    ("q068", "Lợi nhuận gộp theo danh mục sản phẩm từ đơn đã giao là bao nhiêu?", "VI", "SELECT p.category, SUM(o.quantity * (o.unit_price - p.cost_price)) AS gross_margin FROM orders o JOIN products p ON p.product_id = o.product_id WHERE o.status = 'delivered' GROUP BY p.category ORDER BY p.category", "hard", ["join", "group", "arithmetic"]),
    ("q069", "Show customers whose total delivered revenue exceeds 5,000.", "EN", "SELECT c.customer_name, SUM(o.total_amount) AS delivered_revenue FROM customers c JOIN orders o ON o.customer_id = c.customer_id WHERE o.status = 'delivered' GROUP BY c.customer_id, c.customer_name HAVING SUM(o.total_amount) > 5000 ORDER BY delivered_revenue DESC", "hard", ["join", "having"]),
    ("q070", "Khách hàng nào có cả báo giá được chấp nhận và hóa đơn đã thanh toán?", "VI", "SELECT c.customer_name FROM customers c WHERE EXISTS (SELECT 1 FROM quotations q WHERE q.customer_id = c.customer_id AND q.status = 'accepted') AND EXISTS (SELECT 1 FROM invoices i JOIN orders o ON o.order_id = i.order_id WHERE o.customer_id = c.customer_id AND i.status = 'paid') ORDER BY c.customer_id", "hard", ["exists", "join"]),
    ("q071", "What is the month with the highest order revenue?", "EN", "SELECT substr(order_date, 1, 7) AS order_month, SUM(total_amount) AS revenue FROM orders GROUP BY order_month ORDER BY revenue DESC, order_month LIMIT 1", "hard", ["date", "group", "order"]),
    ("q072", "Tổng giá trị đơn hàng của khách hàng Enterprise theo từng tháng là bao nhiêu?", "VI", "SELECT substr(o.order_date, 1, 7) AS order_month, SUM(o.total_amount) AS enterprise_revenue FROM orders o JOIN customers c ON c.customer_id = o.customer_id WHERE c.segment = 'Enterprise' GROUP BY order_month ORDER BY order_month", "hard", ["date", "join", "group"]),
]


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _insert_fixture(connection: sqlite3.Connection) -> None:
    connection.executescript(DDL)
    connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)", CUSTOMERS)
    connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", PRODUCTS)
    connection.executemany("INSERT INTO quotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", QUOTATIONS)
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ORDERS
    )
    connection.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?)", INVOICES)
    connection.executemany("INSERT INTO expenses VALUES (?, ?, ?, ?, ?, ?, ?, ?)", EXPENSES)
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Business fixture violates a foreign-key constraint.")


def build_questions(output_path: str | Path = DEFAULT_QUESTIONS) -> Path:
    """Write the deterministic bilingual question catalog as a JSON array."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": item_id,
            "question": question,
            "language": language,
            "gold_sql": gold_sql,
            "difficulty": difficulty,
            "tags": tags,
            "source": "synthetic_business_v1",
            "db_id": "business",
            "db_path": "business.sqlite",
        }
        for item_id, question, language, gold_sql, difficulty, tags in QUESTION_SPECS
    ]
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def verify_questions(
    db_path: str | Path = DEFAULT_DB,
    questions_path: str | Path = DEFAULT_QUESTIONS,
) -> dict[str, Any]:
    """Execute every gold query and return validation counts."""
    db = Path(db_path).resolve()
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Business question file must contain a non-empty JSON array.")
    connection = sqlite3.connect(db)
    failures: list[dict[str, str]] = []
    try:
        for item in questions:
            try:
                cursor = connection.execute(item["gold_sql"])
                if cursor.description is None:
                    raise ValueError("gold SQL does not return a result set")
                cursor.fetchall()
            except (KeyError, sqlite3.Error, ValueError, TypeError) as error:
                failures.append({"id": str(item.get("id", "?")), "error": str(error)})
    finally:
        connection.close()
    if failures:
        raise ValueError(f"Gold SQL validation failed: {failures[:3]}")
    return {"questions": len(questions), "validated": len(questions), "failures": 0}


def create_database(
    path: str | Path = DEFAULT_DB,
    questions_path: str | Path = DEFAULT_QUESTIONS,
) -> Path:
    """Create the database once, then verify existing databases and gold SQL."""
    db = Path(path).resolve()
    questions = Path(questions_path).resolve()
    if not questions.exists():
        build_questions(questions)
    if db.exists():
        verify_questions(db, questions)
        return db
    db.parent.mkdir(parents=True, exist_ok=True)
    with db.open("xb"):
        connection = _connect(db)
        try:
            _insert_fixture(connection)
            connection.commit()
        finally:
            connection.close()
    verify_questions(db, questions)
    return db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.questions.exists():
        build_questions(args.questions)
    if args.verify_only:
        result = verify_questions(args.path, args.questions)
    else:
        result = {"database": str(create_database(args.path, args.questions)), **verify_questions(args.path, args.questions)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
