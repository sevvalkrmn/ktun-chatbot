"""
KTÜN yemekhane menüsü aracı.

Aylık menü KTÜN sitesinde GÖRSEL olarak yayınlanır (metin değil):
  https://www.ktun.edu.tr/Dosyalar/0/images/<TürkçeAy>.jpg

Bu araç görseli indirir, gpt-4o-mini'nin görsel okuma (vision) yeteneğiyle
ayın tüm menüsünü gün-gün JSON'a çıkarır ve diske cache'ler. Böylece her ay
yalnızca BİR vision çağrısı yapılır; sonraki sorular cache'ten yanıtlanır.
"""
import os
import json
import base64
import datetime

import requests
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL

# Aylık menü görseli yoğun bir takvim tablosu olduğundan, doğru OCR için
# çıkarımda daha güçlü bir vision modeli kullanılır (ayda yalnızca 1 çağrı).
VISION_MODEL = "gpt-4o"

BASE         = "https://www.ktun.edu.tr"
IMG_TMPL     = BASE + "/Dosyalar/0/images/{ay}.jpg"
DUYURU_LIST  = BASE + "/tr/Universite/TumDuyurular"
CACHE_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

TR_AYLAR  = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
TR_GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma",
             "Cumartesi", "Pazar"]


def _ascii_fold(s: str) -> str:
    for a, b in {"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
                 "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
                 "â": "a"}.items():
        s = s.replace(a, b)
    return s


# ── Görsel indirme ───────────────────────────────────────────────────────
def _fetch_image(ay_adi: str):
    """Menü görselini indirir. Önce doğrudan URL, olmazsa duyurudan bulur."""
    for cand in (ay_adi, _ascii_fold(ay_adi)):
        url = IMG_TMPL.format(ay=cand)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                return r.content, url
        except requests.RequestException:
            pass
    return _fetch_image_via_announcement(ay_adi)


def _fetch_image_via_announcement(ay_adi: str):
    """TumDuyurular'dan '<Ay> Ayı Yemek Listesi' duyurusunu bulup görsel src'sini çeker."""
    from bs4 import BeautifulSoup

    r = requests.get(DUYURU_LIST, timeout=20)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")

    detay = None
    for a in soup.find_all("a", href=True):
        txt = a.get_text().lower()
        if "yemek" in txt and _ascii_fold(ay_adi).lower() in _ascii_fold(txt).lower():
            detay = a["href"]
            break
    if not detay:
        raise RuntimeError("Yemek listesi duyurusu bulunamadı")

    detay_url = BASE + detay if detay.startswith("/") else detay
    r2 = requests.get(detay_url, timeout=20)
    r2.encoding = "utf-8"
    soup2 = BeautifulSoup(r2.text, "html.parser")

    for im in soup2.find_all("img", src=True):
        src = im["src"]
        if "Dosyalar" in src and src.lower().endswith((".jpg", ".jpeg", ".png")):
            img_url = BASE + src if src.startswith("/") else src
            rr = requests.get(img_url, timeout=20)
            if rr.status_code == 200:
                return rr.content, img_url
    raise RuntimeError("Duyuruda menü görseli bulunamadı")


# ── Vision ile TEK günün menüsünü çıkar ──────────────────────────────────
# Tüm aylık grid'i transkribe etmek yerine yalnızca sorulan günün hücresini
# okumak, komşu sütun/gün karışmalarını büyük ölçüde önler.
_NO_MENU = "YOK"

def _extract_day(image_bytes: bytes, tarih_str: str, gun_adi: str,
                 ay_adi: str, yil: int) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        f"Bu görsel Konya Teknik Üniversitesi (KTÜN) {ay_adi} {yil} yemekhane "
        "menüsüdür; haftalık takvim tablosudur, sütunlar Pazartesi'den Cuma'ya. "
        f"SADECE {tarih_str} {gun_adi} tarihli hücreyi bul ve o hücredeki çorba, "
        "ana yemek, yardımcı yemek ve tatlı/içeceği (varsa kaloriyi) oku. "
        "O hücrenin hemen sağındaki veya solundaki günlerle KARIŞTIRMA; yalnızca "
        f"üstünde {tarih_str} yazan sütunu kullan. "
        "Yemekleri virgülle ayırıp tek satırda yaz. "
        f"Eğer bu tarih için tabloda yemek yoksa (hafta sonu/tatil/boş hücre) "
        f"sadece '{_NO_MENU}' yaz."
    )

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                               "detail": "high"}},
            ],
        }],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ── Genel API ────────────────────────────────────────────────────────────
def get_day_menu(now: datetime.datetime = None) -> dict:
    """
    Verilen günün (varsayılan: bugün) yemekhane menüsünü döndürür.
    Sonuç ay bazlı cache'e yazılır; aynı gün tekrar sorulursa vision çağrısı yapılmaz.
    """
    now = now or datetime.datetime.now()
    ay_no, yil, gun = now.month, now.year, now.day
    ay_adi    = TR_AYLAR[ay_no - 1]
    gun_adi   = TR_GUNLER[now.weekday()]
    tarih_str = f"{gun:02d}.{ay_no:02d}.{yil}"
    source    = IMG_TMPL.format(ay=ay_adi)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"yemek_{yil}_{ay_no:02d}.json")

    days = {}
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            days = json.load(f)

    if str(gun) in days:
        menu = days[str(gun)]
    else:
        image_bytes, source = _fetch_image(ay_adi)
        menu = _extract_day(image_bytes, tarih_str, gun_adi, ay_adi, yil)
        days[str(gun)] = menu
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(days, f, ensure_ascii=False, indent=2)

    return {
        "ay": ay_adi, "yil": yil, "gun": gun, "gun_adi": gun_adi,
        "tarih": tarih_str, "source": source,
        "menu": None if menu.strip().upper().startswith(_NO_MENU) else menu,
    }
