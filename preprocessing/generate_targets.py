import xarray as xr
import numpy as np
import zarr
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Features veri setini aç
print("Features zarr dosyası açılıyor...")
features = xr.open_zarr("data/features.zarr")

# Random seed ayarla (tutarlı sonuçlar için)
random_seed = 42
np.random.seed(random_seed)

# Hedef veri seti için koordinatları ayarla
targets = xr.Dataset(
    coords={
        "station": features.station,
        "forecast_reference_time": features.forecast_reference_time,
        "t": features.t
    }
)

# İstasyon koordinat bilgilerini kopyala
for coord in ["elevation", "longitude", "latitude", "model_height_difference", "owner_id"]:
    if coord in features.coords:
        targets = targets.assign_coords({coord: features[coord]})

# Boyutları tanımla
dims = ["station", "forecast_reference_time", "t"]

# August-Roche-Magnus denklemi sabitleri (makaleden)
a_pos = 17.368
b_pos = 238.83
c_pos = 6.107  # hPa

a_neg = 17.856
b_neg = 245.52
c_neg = 6.108  # hPa

print("Hava sıcaklığı oluşturuluyor...")
# 1. Sıcaklık (T)
shape = (len(targets.station), len(targets.forecast_reference_time), len(targets.t))
temperature = np.zeros(shape)

if "coe_air_temperature_ensavg" in features:
    # Ensemble ortalaması + gerçekçi hata
    base_temp = features["coe_air_temperature_ensavg"].values

    # Zamansal korelasyonu korumak için gaussian smoothing uygula
    error = np.random.normal(0, 1.5, size=base_temp.shape)
    for i in range(len(features.station)):
        for j in range(len(features.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=2)

    # Sıcaklık değerlerini hesapla
    temperature = base_temp + error

    # Gerçekçi sınırları uygula (İsviçre için)
    temperature = np.clip(temperature, -30, 40)
else:
    # Eğer özellik yoksa, gerçekçi değerler oluştur
    # Mevsimselliği ekle - İsviçre için ortalama sıcaklık mevsimsel değişimi
    time_idx = np.linspace(0, 2 * np.pi, len(targets.forecast_reference_time))
    seasonal_temp = 10 - 10 * np.cos(time_idx)  # Kış: ~0°C, Yaz: ~20°C

    # İstasyon yüksekliklerine göre ayarla (her 100m için -0.65°C)
    if "elevation" in features.coords:
        # DataArray'i numpy array'e dönüştür
        station_temp_base = 15 - 0.0065 * features.elevation.values
    else:
        station_temp_base = np.ones(len(targets.station)) * 15

    # Base temperature oluştur
    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            # Numpy array'leri kullan ve boyut hatalarını önle
            temperature[i, :, j] = station_temp_base[i] + seasonal_temp

    # Random varyasyon ekle (zamansal korelasyonlu)
    error = np.random.normal(0, 3, size=shape)
    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=2)

    temperature = temperature + error
    temperature = np.clip(temperature, -30, 40)  # Gerçekçi sınırlar

# Hedef değişkene ekle
targets["obs_air_temperature"] = (dims, temperature)

print("Yüzey hava basıncı oluşturuluyor...")
# 2. Yüzey Hava Basıncı (P)
pressure = np.zeros(shape)

if "coe_surface_air_pressure_ensavg" in features:
    base_pressure = features["coe_surface_air_pressure_ensavg"].values

    # Gerçekçi basınç hatası (zamansal korelasyonlu)
    error = np.random.normal(0, 1.5, size=base_pressure.shape)
    for i in range(len(features.station)):
        for j in range(len(features.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=3)

    pressure = base_pressure + error
else:
    # Deniz seviyesi standart basıncı
    sea_level_pressure = 1013.25  # hPa

    # İstasyon yüksekliğine göre basıncı ayarla
    if "elevation" in features.coords:
        # Barometrik yükseklik formülü - numpy array kullan
        station_pressure_base = sea_level_pressure * np.exp(-features.elevation.values / 8000)
    else:
        station_pressure_base = np.ones(len(targets.station)) * sea_level_pressure

    # Zamansal değişim ekle (mevsimsel ve günlük)
    # Mevsimsel değişim (kış aylarında daha yüksek basınç)
    time_idx = np.linspace(0, 2 * np.pi, len(targets.forecast_reference_time))
    seasonal_pressure = 5 * np.cos(time_idx)  # ±5 hPa mevsimsel varyasyon

    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            pressure[i, :, j] = station_pressure_base[i] + seasonal_pressure

    # Random varyasyon ekle (zamansal korelasyonlu)
    error = np.random.normal(0, 3, size=shape)
    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=3)

    pressure = pressure + error

# Gerçekçi basınç sınırları uygula (İsviçre'deki yüksek dağlar için)
pressure = np.clip(pressure, 500, 1100)
targets["obs_surface_air_pressure"] = (dims, pressure)

print("Bağıl nem oluşturuluyor...")
# 3. Bağıl Nem (RH)
rh = np.zeros(shape)

if "coe_relative_humidity_ensavg" in features:
    base_rh = features["coe_relative_humidity_ensavg"].values

    # Gerçekçi nem hatası (zamansal korelasyonlu)
    error = np.random.normal(0, 5, size=base_rh.shape)
    for i in range(len(features.station)):
        for j in range(len(features.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=2)

    rh = base_rh + error
else:
    # İstasyon ve mevsim bazlı nem - numpy arrays kullan
    # Mevsimsel nem değişimi (yaz aylarında daha düşük nem)
    time_idx = np.linspace(0, 2 * np.pi, len(targets.forecast_reference_time))
    seasonal_rh = 60 + 15 * np.cos(time_idx)  # Kış: ~75%, Yaz: ~45%

    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            rh[i, :, j] = seasonal_rh

    # Random varyasyon ekle (zamansal korelasyonlu)
    error = np.random.normal(0, 10, size=shape)
    for i in range(len(targets.station)):
        for j in range(len(targets.t)):
            error[i, :, j] = gaussian_filter1d(error[i, :, j], sigma=2)

    rh = rh + error

# Gerçekçi nem sınırları uygula
rh = np.clip(rh, 0, 100)
targets["obs_relative_humidity"] = (dims, rh)

print("Çiy noktası sıcaklığı hesaplanıyor...")


# 4. Çiy Noktası (Td) - Makaledeki formüllere göre hesaplama
def calculate_dew_point(T, RH):
    """
    Makalede belirtilen formüllere göre çiy noktası hesabı:
    - T: Sıcaklık (°C)
    - RH: Bağıl Nem (%)
    """
    # Sıcaklığa göre sabitleri belirle
    a = np.where(T >= 0, a_pos, a_neg)
    b = np.where(T >= 0, b_pos, b_neg)
    c = np.where(T >= 0, c_pos, c_neg)

    # Doymuş buhar basıncı (es)
    es = c * np.exp((a * T) / (b + T))

    # Mevcut buhar basıncı (e)
    e = (RH / 100) * es

    # Çiy noktası hesabı için güvenli değerler
    safe_e = np.maximum(e, 1e-6)  # Çok küçük değerlerden kaçınmak için

    # T >= 0 için çiy noktası hesabı
    td_pos = b_pos * np.log(safe_e / c_pos) / (a_pos - np.log(safe_e / c_pos))

    # T < 0 için çiy noktası hesabı
    td_neg = b_neg * np.log(safe_e / c_neg) / (a_neg - np.log(safe_e / c_neg))

    # Sıcaklığa göre ilgili formülü seç
    td = np.where(T >= 0, td_pos, td_neg)

    # Çiy noktası sıcaklıktan yüksek olamaz
    td = np.minimum(td, T)

    return td


# Çiy noktası hesapla
T = targets["obs_air_temperature"].values
RH = targets["obs_relative_humidity"].values
dew_point = calculate_dew_point(T, RH)

# Çiy noktasını ekle
targets["obs_dew_point_temperature"] = (dims, dew_point)

print("Su buharı karışım oranı hesaplanıyor...")


# 5. Su buharı karışım oranı (r) - Makaledeki formüllere göre hesaplama
def calculate_mixing_ratio(T, Td, P):
    """
    Makalede belirtilen formüllere göre su buharı karışım oranı hesabı:
    - T: Sıcaklık (°C)
    - Td: Çiy Noktası (°C)
    - P: Basınç (hPa)
    """
    # Td için sabitleri belirle
    a_td = np.where(T >= 0, a_pos, a_neg)
    b_td = np.where(T >= 0, b_pos, b_neg)
    c_td = np.where(T >= 0, c_pos, c_neg)

    # Buhar basıncı (e) - Makaledeki Eşitlik 5
    e = c_td * np.exp((a_td * Td) / (b_td + Td))

    # Karışım oranı (g/kg) - Makaledeki Eşitlik 7
    # Sayısal kararlılık için paydada küçük değer ekliyoruz
    r = 622.0 * e / (P - e + 1e-6)

    return r


# Su buharı karışım oranını hesapla
P = targets["obs_surface_air_pressure"].values
Td = targets["obs_dew_point_temperature"].values
mixing_ratio = calculate_mixing_ratio(T, Td, P)

# Gerçekçi sınırları uygula
mixing_ratio = np.clip(mixing_ratio, 0, 30)  # g/kg
targets["obs_water_vapor_mixing_ratio"] = (dims, mixing_ratio)

print("Kalite kontrol yapılıyor...")


# Rassal bazı değerleri NaN ile değiştirerek gerçek gözlemleri taklit et
def add_missing_values(data, missing_rate=0.005):
    """Verilerin belirli bir yüzdesini NaN ile değiştirir"""
    mask = np.random.random(data.shape) < missing_rate
    result = data.copy()
    result[mask] = np.nan
    return result


# Her değişken için rastgele eksik değerler ekle (%0.5)
for var in targets.data_vars:
    targets[var].values = add_missing_values(targets[var].values)

# Kalite kontrolü yap
# 1. Çiy noktası sıcaklıktan yüksek olmamalı
mask = targets["obs_dew_point_temperature"].values > targets["obs_air_temperature"].values
targets["obs_dew_point_temperature"].values[mask] = targets["obs_air_temperature"].values[mask]

# 2. Bağıl nem 0-100 arasında olmalı
targets["obs_relative_humidity"].values = np.clip(targets["obs_relative_humidity"].values, 0, 100)

# 3. Su buharı karışım oranı pozitif olmalı
targets["obs_water_vapor_mixing_ratio"].values = np.maximum(targets["obs_water_vapor_mixing_ratio"].values, 0)

# İstatistikleri göster
print("\nOluşturulan verilerin istatistikleri:")
for var in targets.data_vars:
    data = targets[var].values
    valid_data = data[~np.isnan(data)]
    print(f"{var}:")
    print(f"  Min: {valid_data.min():.2f}, Max: {valid_data.max():.2f}")
    print(f"  Mean: {valid_data.mean():.2f}, Std: {valid_data.std():.2f}")
    print(f"  Missing values: {np.isnan(data).sum()} ({np.isnan(data).sum() / data.size * 100:.2f}%)")

print("\nHedef verisi kaydediliyor...")
# Veriyi kaydet
output_path = "data/targets.zarr"
targets.to_zarr(output_path, mode="w")

print(f"\nHedef verisi başarıyla oluşturuldu ve '{output_path}' konumuna kaydedildi!")
print(f"Toplam boyutlar: {targets.dims}")
print(f"Değişkenler: {list(targets.data_vars.keys())}")