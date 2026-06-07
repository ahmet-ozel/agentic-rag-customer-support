# Katkıda Bulunma Rehberi

AgentDesk projesine katkıda bulunmak istediğiniz için teşekkürler! Bu rehber, katkı sürecini kolaylaştırmak için hazırlanmıştır.

## İçindekiler

- [Davranış Kuralları](#davranış-kuralları)
- [Nasıl Katkıda Bulunabilirim?](#nasıl-katkıda-bulunabilirim)
- [Geliştirme Ortamı Kurulumu](#geliştirme-ortamı-kurulumu)
- [Kod Standartları](#kod-standartları)
- [Test Yazma](#test-yazma)
- [Pull Request Süreci](#pull-request-süreci)
- [Issue Açma](#issue-açma)
- [Proje Yapısı](#proje-yapısı)

---

## Davranış Kuralları

Bu proje, saygılı ve kapsayıcı bir ortam sağlamayı taahhüt eder. Tüm katılımcılardan:

- Saygılı ve yapıcı iletişim kurmaları
- Farklı bakış açılarına açık olmaları
- Yapıcı geri bildirim vermeleri ve kabul etmeleri
- Topluluk için en iyisine odaklanmaları

beklenir.

---

## Nasıl Katkıda Bulunabilirim?

### Hata Bildirimi

1. Önce [mevcut issue'ları](../../issues) kontrol edin  -  aynı hata zaten bildirilmiş olabilir
2. Yeni bir issue açın ve şunları ekleyin:
   - Hatanın net açıklaması
   - Hatayı yeniden üretme adımları
   - Beklenen davranış vs. gerçekleşen davranış
   - Python versiyonu, işletim sistemi bilgisi
   - Varsa hata logları

### Özellik Önerisi

1. Bir issue açarak önerinizi açıklayın
2. Kullanım senaryosunu ve motivasyonu belirtin
3. Mümkünse tasarım önerisi ekleyin

### Kod Katkısı

1. Repo'yu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/ozellik-adi`)
3. Değişikliklerinizi yapın
4. Testleri yazın ve çalıştırın
5. Pull request açın

---

## Geliştirme Ortamı Kurulumu

### Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- Git

### Adımlar

```bash
# 1. Repo'yu fork edin ve klonlayın
git clone https://github.com/YOUR_USERNAME/agentdesk.git
cd agentdesk

# 2. Sanal ortam oluşturun
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Ortam değişkenlerini ayarlayın
cp .env.example .env
# .env dosyasını düzenleyin

# 5. Testlerin geçtiğini doğrulayın
pytest tests/ -v
```

### Servisleri Başlatma (Opsiyonel)

Entegrasyon testleri veya tam geliştirme için:

```bash
docker compose up -d postgres qdrant
```

---

## Kod Standartları

### Python Stili

- **Linter:** [Ruff](https://docs.astral.sh/ruff/) kullanılır
- **Type Hints:** Tüm fonksiyon imzalarında type hint kullanın
- **Docstrings:** Tüm public sınıf ve fonksiyonlarda docstring yazın
- **Import Sırası:** `from __future__ import annotations` her dosyanın başında

```bash
# Lint kontrolü
make lint
# veya
ruff check src/ tests/
```

### Commit Mesajları

[Conventional Commits](https://www.conventionalcommits.org/) formatını kullanın:

```
feat: yeni özellik açıklaması
fix: hata düzeltme açıklaması
docs: dokümantasyon değişikliği
test: test ekleme veya düzeltme
refactor: kod yeniden yapılandırma
chore: bakım işleri (bağımlılık güncelleme vb.)
```

Örnekler:

```
feat: Milvus vektör deposu adaptörü eklendi
fix: SessionManager TTL hesaplaması düzeltildi
docs: API referansı güncellendi
test: ChunkingEngine semantic strateji testleri eklendi
```

### Branch İsimlendirme

```
feature/ozellik-adi
fix/hata-aciklamasi
docs/dokuman-guncelleme
test/test-ekleme
```

---

## Test Yazma

### Test Çalıştırma

```bash
# Tüm testler
make test

# Sadece birim testleri
make test-unit

# Sadece entegrasyon testleri
make test-int

# Belirli bir test dosyası
pytest tests/unit/test_config.py -v

# Belirli bir test sınıfı
pytest tests/unit/test_config.py::TestConfigManagerLoad -v
```

### Test Yazma Kuralları

1. **Her yeni özellik için test yazın**  -  PR'lar test olmadan kabul edilmez
2. **Birim testleri** `tests/unit/` altına, **entegrasyon testleri** `tests/integration/` altına yazın
3. **Mock kullanın**  -  harici servislere (LLM, veritabanı) bağımlılık oluşturmayın
4. **Açıklayıcı test isimleri** kullanın: `test_upload_document_returns_completed_status`
5. **Edge case'leri** test edin: boş girdi, geçersiz parametre, zaman aşımı

### Test Yapısı Örneği

```python
"""Unit tests for YeniModul."""

from __future__ import annotations

import pytest

from src.yeni_modul import YeniSinif


class TestYeniOzellik:
    def test_basarili_senaryo(self) -> None:
        """Normal kullanımda beklenen sonucu döndürür."""
        obj = YeniSinif()
        result = obj.islem("girdi")
        assert result == "beklenen"

    def test_bos_girdi(self) -> None:
        """Boş girdi için uygun hata döndürür."""
        obj = YeniSinif()
        with pytest.raises(ValueError):
            obj.islem("")

    @pytest.mark.asyncio
    async def test_async_islem(self) -> None:
        """Async işlem doğru sonuç döndürür."""
        obj = YeniSinif()
        result = await obj.async_islem()
        assert result is not None
```

---

## Pull Request Süreci

### PR Açmadan Önce

1.  Tüm testler geçiyor: `pytest tests/ -v`
2.  Lint hataları yok: `ruff check src/ tests/`
3.  Yeni özellik için testler yazıldı
4.  Docstring'ler eklendi
5.  `config.yaml` değişiklikleri varsa dokümante edildi

### PR Şablonu

```markdown
## Açıklama
[Değişikliğin kısa açıklaması]

## Değişiklik Türü
- [ ] Hata düzeltme (bug fix)
- [ ] Yeni özellik (feature)
- [ ] Kırıcı değişiklik (breaking change)
- [ ] Dokümantasyon

## Test
- [ ] Mevcut testler geçiyor
- [ ] Yeni testler eklendi

## İlgili Issue
Closes #[issue numarası]
```

### Review Süreci

1. PR açıldığında otomatik testler çalışır
2. En az bir maintainer review'u gereklidir
3. Tüm review yorumları çözülmelidir
4. Merge öncesi tüm testler geçmelidir

---

## Issue Açma

### Hata Raporu Şablonu

```markdown
## Hata Açıklaması
[Hatanın net açıklaması]

## Yeniden Üretme Adımları
1. ...
2. ...
3. ...

## Beklenen Davranış
[Ne olması gerektiği]

## Gerçekleşen Davranış
[Ne olduğu]

## Ortam
- Python: [versiyon]
- OS: [işletim sistemi]
- Docker: [versiyon, varsa]

## Loglar
```
[Hata logları]
```
```

### Özellik Önerisi Şablonu

```markdown
## Özellik Açıklaması
[Önerilen özelliğin açıklaması]

## Motivasyon
[Neden bu özelliğe ihtiyaç var]

## Kullanım Senaryosu
[Nasıl kullanılacağı]

## Tasarım Önerisi (Opsiyonel)
[Teknik yaklaşım önerisi]
```

---

## Proje Yapısı

Yeni dosya eklerken bu yapıyı takip edin:

```
src/
├── agent/          # Agent loop ve ilgili mantık
├── api/            # FastAPI endpoint'leri (her endpoint ayrı dosya)
├── chunking/       # Doküman parçalama stratejileri
├── config/         # Yapılandırma yönetimi
├── llm/            # LLM istemci katmanı
├── mcp/            # MCP sunucu yönetimi
├── models/         # Pydantic şemaları
├── router/         # Niyet yönlendirme
├── session/        # Oturum yönetimi
├── store/          # Reference store
└── vectorstore/    # Vektör deposu adaptörleri
```

### Yeni Bileşen Ekleme

1. `src/` altında uygun dizinde modül oluşturun
2. `__init__.py` dosyasını güncelleyin
3. Gerekirse `config.yaml`'a yapılandırma bölümü ekleyin
4. `src/config/models.py`'ye Pydantic modeli ekleyin
5. Birim testlerini `tests/unit/` altına yazın
6. README'yi güncelleyin

---

## Yardım

Sorularınız için:
- [GitHub Issues](../../issues) üzerinden soru sorabilirsiniz
- Mevcut [tartışmaları](../../discussions) inceleyebilirsiniz

Katkılarınız için teşekkürler!
