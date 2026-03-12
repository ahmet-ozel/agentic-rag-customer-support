# postgres-mcp — PostgreSQL MCP Sunucusu

PostgreSQL müşteri veritabanına **salt okunur** erişim sağlayan MCP (Model Context Protocol) sunucusu.

## Güvenlik

- **Salt okunur erişim:** Yalnızca `SELECT` sorguları kabul edilir. `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE` ve diğer yazma işlemleri reddedilir.
- **Parametreli sorgular:** Tüm kullanıcı girdileri parametreli sorgular ile işlenir — SQL injection koruması sağlar.
- **Sorgu zaman aşımı:** Uzun süren sorgular 30 saniye sonra sonlandırılır.
- **Tablo kısıtlaması:** Yalnızca `allowed_tables` listesindeki tablolara erişim izni verilir.

## Ortam Değişkenleri

| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `POSTGRES_HOST` | PostgreSQL sunucu adresi | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port numarası | `5432` |
| `POSTGRES_DB` | Veritabanı adı | `agentdesk` |
| `POSTGRES_READONLY_USER` | Salt okunur kullanıcı adı | `agentdesk_readonly` |
| `POSTGRES_READONLY_PASSWORD` | Salt okunur kullanıcı şifresi | *(zorunlu)* |

## Veritabanı Kullanıcısı Oluşturma

Üretim ortamında salt okunur bir PostgreSQL kullanıcısı oluşturun:

```sql
CREATE USER agentdesk_readonly WITH PASSWORD 'güvenli_şifre';
GRANT CONNECT ON DATABASE agentdesk TO agentdesk_readonly;
GRANT USAGE ON SCHEMA public TO agentdesk_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agentdesk_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agentdesk_readonly;
```

## Araçlar

| Araç | Açıklama |
|------|----------|
| `query_database` | Parametreli SELECT sorgusu çalıştırır |
| `list_tables` | Erişilebilir tabloları listeler |
| `describe_table` | Tablo sütun yapısını döndürür |

## config.yaml Entegrasyonu

```yaml
mcp_servers:
  postgres-mcp:
    enabled: true
    transport: stdio
    command: "python"
    args: ["-m", "mcp_servers.postgres_mcp.server"]
    env:
      POSTGRES_HOST: "${POSTGRES_HOST}"
      POSTGRES_PORT: "${POSTGRES_PORT}"
      POSTGRES_DB: "${POSTGRES_DB}"
      POSTGRES_READONLY_USER: "${POSTGRES_READONLY_USER}"
      POSTGRES_READONLY_PASSWORD: "${POSTGRES_READONLY_PASSWORD}"
```
