-- ============================================================================
-- AgentDesk RAG Platform - Örnek (Seed) Veriler
-- Açıklama: Geliştirme ve test ortamı için gerçekçi örnek veriler
-- ============================================================================

-- Planlar
INSERT INTO plans (name, price, features, max_users) VALUES
('Free', 0.00, '{"storage": "1GB", "support": "community", "api_calls": 100}', 1),
('Starter', 29.99, '{"storage": "10GB", "support": "email", "api_calls": 5000, "custom_branding": false}', 5),
('Professional', 99.99, '{"storage": "100GB", "support": "priority", "api_calls": 50000, "custom_branding": true, "analytics": true}', 25),
('Enterprise', 299.99, '{"storage": "unlimited", "support": "dedicated", "api_calls": "unlimited", "custom_branding": true, "analytics": true, "sla": "99.9%"}', NULL);

-- Müşteriler (placeholder Türkçe isimler)
INSERT INTO customers (name, email, phone, company, plan_id, status, created_at) VALUES
('[Ad Soyad 1]', '[email1]@example.com', '+90 532 111 2233', 'Teknoloji A.Ş.', 3, 'active', '2024-01-15 10:30:00'),
('[Ad Soyad 2]', '[email2]@example.com', '+90 544 222 3344', 'Yazılım Ltd.', 2, 'active', '2024-02-20 14:00:00'),
('[Ad Soyad 3]', '[email3]@example.com', '+90 555 333 4455', 'Danışmanlık Hizmetleri', 4, 'active', '2024-03-01 09:15:00'),
('[Ad Soyad 4]', '[email4]@example.com', '+90 533 444 5566', NULL, 1, 'active', '2024-04-10 16:45:00'),
('[Ad Soyad 5]', '[email5]@example.com', '+90 542 555 6677', 'Medya Grubu', 3, 'suspended', '2024-01-05 11:00:00'),
('[Ad Soyad 6]', '[email6]@example.com', '+90 537 666 7788', 'E-Ticaret A.Ş.', 2, 'cancelled', '2024-05-12 08:30:00');

-- Abonelikler
INSERT INTO subscriptions (customer_id, plan_id, start_date, end_date, status) VALUES
(1, 3, '2024-01-15', '2025-01-15', 'active'),
(2, 2, '2024-02-20', '2025-02-20', 'active'),
(3, 4, '2024-03-01', NULL, 'active'),
(4, 1, '2024-04-10', '2025-04-10', 'active'),
(5, 3, '2024-01-05', '2025-01-05', 'cancelled'),
(6, 2, '2024-05-12', '2024-11-12', 'cancelled');

-- Destek talepleri (çeşitli durum ve öncelikler)
INSERT INTO support_tickets (customer_id, subject, description, status, priority, created_at, updated_at) VALUES
(1, 'API entegrasyonu hakkında soru', 'REST API ile entegrasyon yaparken 401 hatası alıyorum. API anahtarımı doğru kullandığımdan eminim.', 'open', 'high', '2024-06-01 09:00:00', '2024-06-01 09:00:00'),
(1, 'Fatura detayı talebi', 'Mayıs ayı faturamın detaylı dökümünü alabilir miyim?', 'resolved', 'low', '2024-05-20 14:30:00', '2024-05-21 10:00:00'),
(2, 'Plan yükseltme bilgisi', 'Starter plandan Professional plana geçiş yapmak istiyorum. Mevcut verilerim korunacak mı?', 'in_progress', 'medium', '2024-06-05 11:15:00', '2024-06-06 08:00:00'),
(3, 'Özel entegrasyon desteği', 'Enterprise planımız kapsamında SAP entegrasyonu için teknik destek talep ediyoruz.', 'open', 'critical', '2024-06-10 08:00:00', '2024-06-10 08:00:00'),
(3, 'SLA raporu talebi', 'Son 3 aylık SLA performans raporunu paylaşabilir misiniz?', 'resolved', 'medium', '2024-05-15 16:00:00', '2024-05-16 09:30:00'),
(4, 'Ücretsiz plan limitleri', 'Free plandaki API çağrı limitini aştım. Geçici olarak limit artırılabilir mi?', 'open', 'medium', '2024-06-12 10:00:00', '2024-06-12 10:00:00'),
(5, 'Hesap yeniden etkinleştirme', 'Askıya alınan hesabımı yeniden etkinleştirmek istiyorum. Ödeme sorununu çözdüm.', 'in_progress', 'high', '2024-06-08 13:00:00', '2024-06-09 11:00:00'),
(2, 'Veri dışa aktarma', 'Tüm verilerimi CSV formatında dışa aktarmak istiyorum.', 'closed', 'low', '2024-04-25 09:45:00', '2024-04-26 14:00:00'),
(6, 'Hesap silme talebi', 'İptal edilen hesabımdaki tüm verilerin silinmesini talep ediyorum.', 'open', 'medium', '2024-06-15 07:30:00', '2024-06-15 07:30:00'),
(1, 'Webhook yapılandırması', 'Webhook bildirimleri için endpoint yapılandırması nasıl yapılır?', 'resolved', 'low', '2024-05-10 15:00:00', '2024-05-11 11:30:00');

-- Faturalar (çeşitli durumlar)
INSERT INTO invoices (customer_id, amount, status, due_date, paid_at, created_at) VALUES
(1, 99.99, 'paid', '2024-06-15', '2024-06-10 09:00:00', '2024-06-01 00:00:00'),
(2, 29.99, 'paid', '2024-06-20', '2024-06-18 14:30:00', '2024-06-01 00:00:00'),
(3, 299.99, 'paid', '2024-06-01', '2024-05-28 10:00:00', '2024-05-15 00:00:00'),
(4, 0.00, 'paid', '2024-06-10', '2024-06-10 00:00:00', '2024-06-01 00:00:00'),
(5, 99.99, 'overdue', '2024-05-15', NULL, '2024-05-01 00:00:00'),
(1, 99.99, 'pending', '2024-07-15', NULL, '2024-07-01 00:00:00');

-- Ürünler
INSERT INTO products (name, description, price, category) VALUES
('AgentDesk Starter Paketi', 'Küçük işletmeler için temel müşteri destek çözümü', 29.99, 'paket'),
('AgentDesk Pro Paketi', 'Orta ölçekli işletmeler için gelişmiş destek platformu', 99.99, 'paket'),
('API Eklentisi', 'REST API erişimi ve webhook desteği', 19.99, 'eklenti'),
('Özel Entegrasyon Hizmeti', 'SAP, Salesforce ve diğer sistemlerle entegrasyon', 499.99, 'hizmet'),
('Eğitim Paketi', 'Platform kullanımı için kapsamlı eğitim programı', 149.99, 'hizmet'),
('Ek Depolama (100GB)', 'Ek bulut depolama alanı', 9.99, 'eklenti');

-- SSS kategorileri (üst-alt hiyerarşi)
INSERT INTO faq_categories (name, description, parent_id) VALUES
('Genel', 'Genel sorular ve platform hakkında bilgiler', NULL),
('Teknik Destek', 'Teknik sorunlar ve çözümleri', NULL),
('Faturalandırma', 'Ödeme ve fatura ile ilgili sorular', NULL),
('API ve Entegrasyon', 'API kullanımı ve üçüncü parti entegrasyonlar', 2),
('Hesap Yönetimi', 'Hesap ayarları ve kullanıcı yönetimi', 1),
('Plan ve Fiyatlandırma', 'Plan karşılaştırma ve fiyat bilgileri', 3),
('Veri Güvenliği', 'Veri koruma ve güvenlik politikaları', 2);

-- Aktivite logu
INSERT INTO activity_log (customer_id, action, details, created_at) VALUES
(1, 'login', '{"ip": "192.168.1.10", "device": "Chrome/Windows"}', '2024-06-15 08:00:00'),
(1, 'api_call', '{"endpoint": "/api/v1/chat", "method": "POST", "status": 200}', '2024-06-15 08:05:00'),
(2, 'login', '{"ip": "10.0.0.5", "device": "Safari/macOS"}', '2024-06-15 09:30:00'),
(2, 'document_upload', '{"filename": "kullanim_kilavuzu.pdf", "size_mb": 2.5}', '2024-06-15 09:35:00'),
(3, 'plan_change', '{"from": "Professional", "to": "Enterprise"}', '2024-03-01 09:15:00'),
(4, 'login', '{"ip": "172.16.0.1", "device": "Firefox/Linux"}', '2024-06-14 17:00:00'),
(5, 'account_suspended', '{"reason": "payment_overdue", "invoice_id": 5}', '2024-06-01 00:00:00'),
(1, 'settings_update', '{"field": "notification_preferences", "value": "email_only"}', '2024-06-14 12:00:00');
