-- ============================================================================
-- AgentDesk RAG Platform - Veritabanı Şeması
-- Migration: 001_initial_schema.sql
-- Açıklama: Müşteri destek platformu için temel tablo yapısı
-- ============================================================================

-- Plan tablosu (müşteri tablosundan önce oluşturulmalı - FK bağımlılığı)
CREATE TABLE IF NOT EXISTS plans (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    features JSONB,
    max_users INTEGER
);

-- Müşteri tablosu
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    company VARCHAR(100),
    plan_id INTEGER REFERENCES plans(id),
    status VARCHAR(20) DEFAULT 'active',       -- active, suspended, cancelled
    created_at TIMESTAMP DEFAULT NOW()
);

-- Abonelik tablosu
CREATE TABLE IF NOT EXISTS subscriptions (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    plan_id INTEGER REFERENCES plans(id),
    start_date DATE NOT NULL,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'active'        -- active, cancelled, expired
);

-- Destek talepleri
CREATE TABLE IF NOT EXISTS support_tickets (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    subject VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'open',         -- open, in_progress, resolved, closed
    priority VARCHAR(10) DEFAULT 'medium',     -- low, medium, high, critical
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Faturalar
CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',      -- pending, paid, overdue
    due_date DATE NOT NULL,
    paid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Aktivite logu
CREATE TABLE IF NOT EXISTS activity_log (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Ürünler
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10,2),
    category VARCHAR(50)
);

-- SSS (Sıkça Sorulan Sorular) kategorileri
CREATE TABLE IF NOT EXISTS faq_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_id INTEGER REFERENCES faq_categories(id)
);

-- ============================================================================
-- İndeksler - Sık sorgulanan sütunlar için performans optimizasyonu
-- ============================================================================

-- Müşteri tablosu indeksleri
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status);
CREATE INDEX IF NOT EXISTS idx_customers_plan_id ON customers(plan_id);

-- Abonelik tablosu indeksleri
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id ON subscriptions(customer_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);

-- Destek talepleri indeksleri
CREATE INDEX IF NOT EXISTS idx_support_tickets_customer_id ON support_tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_support_tickets_status ON support_tickets(status);
CREATE INDEX IF NOT EXISTS idx_support_tickets_priority ON support_tickets(priority);

-- Fatura indeksleri
CREATE INDEX IF NOT EXISTS idx_invoices_customer_id ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_due_date ON invoices(due_date);

-- Aktivite logu indeksleri
CREATE INDEX IF NOT EXISTS idx_activity_log_customer_id ON activity_log(customer_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_created_at ON activity_log(created_at);

-- Ürün indeksleri
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- SSS kategori indeksleri
CREATE INDEX IF NOT EXISTS idx_faq_categories_parent_id ON faq_categories(parent_id);
