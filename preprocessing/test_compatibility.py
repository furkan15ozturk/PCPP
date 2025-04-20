import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Veri setlerini yükle
print("Veri setleri yükleniyor...")
features = xr.open_zarr("data/features.zarr")
targets = xr.open_zarr("data/targets.zarr")

print("\n" + "=" * 80)
print("UYUMLULUK TESTİ RAPORU")
print("=" * 80)

# 1. Boyut Uyumluluğu Testi
print("\n1. BOYUT UYUMLULUĞU TESTİ")
print("-" * 50)

matching_dims = {}
for dim in ['station', 'forecast_reference_time', 't']:
    if dim in features.dims and dim in targets.dims:
        features_len = len(features[dim])
        targets_len = len(targets[dim])
        match = features_len == targets_len
        matching_dims[dim] = match
        print(f"Boyut '{dim}': Features ({features_len}) - Targets ({targets_len}) - Eşleşme: {match}")
    else:
        missing = []
        if dim not in features.dims:
            missing.append("Features")
        if dim not in targets.dims:
            missing.append("Targets")
        print(f"Boyut '{dim}' şurada eksik: {', '.join(missing)}")

all_dims_match = all(matching_dims.values())
print(f"\nTüm boyutlar eşleşiyor mu? {'✓ EVET' if all_dims_match else '✗ HAYIR'}")

# 2. Koordinat Uyumluluğu Testi
print("\n2. KOORDİNAT UYUMLULUĞU TESTİ")
print("-" * 50)

important_coords = ['longitude', 'latitude', 'elevation', 'model_height_difference']
matching_coords = {}

for coord in important_coords:
    if coord in features.coords and coord in targets.coords:
        # Numpy dizilerine dönüştür ve karşılaştır
        features_values = features[coord].values
        targets_values = targets[coord].values

        # Dizilerin şekli aynı mı kontrol et
        if features_values.shape == targets_values.shape:
            # Tüm değerler aynı mı kontrol et
            is_equal = np.array_equal(features_values, targets_values)
            matching_coords[coord] = is_equal
            print(f"Koordinat '{coord}': {'✓ Eşleşiyor' if is_equal else '✗ Eşleşmiyor'}")
        else:
            matching_coords[coord] = False
            print(
                f"Koordinat '{coord}': ✗ Boyut uyumsuzluğu - Features {features_values.shape}, Targets {targets_values.shape}")
    else:
        missing = []
        if coord not in features.coords:
            missing.append("Features")
        if coord not in targets.coords:
            missing.append("Targets")
        matching_coords[coord] = False
        print(f"Koordinat '{coord}' şurada eksik: {', '.join(missing)}")

all_coords_match = all(matching_coords.values())
print(f"\nTüm önemli koordinatlar eşleşiyor mu? {'✓ EVET' if all_coords_match else '✗ HAYIR'}")

# 3. Değişken Tutarlılığı Testi
print("\n3. DEĞİŞKEN TUTARLILIĞI TESTİ")
print("-" * 50)

# Değişken eşleştirmeleri (makalede belirtilen değişkenler)
var_mappings = {
    'air_temperature': ('coe:air_temperature_ensavg', 'obs:air_temperature'),
    'dew_point_temperature': ('coe:dew_point_temperature_ensavg', 'obs:dew_point_temperature'),
    'surface_air_pressure': ('coe:surface_air_pressure_ensavg', 'obs:surface_air_pressure'),
    'relative_humidity': ('coe:relative_humidity_ensavg', 'obs:relative_humidity'),
    'water_vapor_mixing_ratio': ('coe:water_vapor_mixing_ratio_ensavg', 'obs:water_vapor_mixing_ratio')
}

for key, (feature_var, target_var) in var_mappings.items():
    print(f"\nDeğişken: {key}")

    # Features ve targets'da değişkenin varlığını kontrol et
    feature_exists = feature_var in features
    target_exists = target_var in targets

    if feature_exists and target_exists:
        # İstatistiksel özet
        feature_stats = {
            'min': float(features[feature_var].min().values),
            'max': float(features[feature_var].max().values),
            'mean': float(features[feature_var].mean().values),
            'std': float(features[feature_var].std().values)
        }

        target_stats = {
            'min': float(targets[target_var].min().values),
            'max': float(targets[target_var].max().values),
            'mean': float(targets[target_var].mean().values),
            'std': float(targets[target_var].std().values)
        }

        print(f"  Features ({feature_var}):")
        print(f"    Min: {feature_stats['min']:.2f}, Max: {feature_stats['max']:.2f}")
        print(f"    Mean: {feature_stats['mean']:.2f}, Std: {feature_stats['std']:.2f}")

        print(f"  Targets ({target_var}):")
        print(f"    Min: {target_stats['min']:.2f}, Max: {target_stats['max']:.2f}")
        print(f"    Mean: {target_stats['mean']:.2f}, Std: {target_stats['std']:.2f}")

        # Değer aralığı makul mu?
        reasonable_range = True

        if key == 'air_temperature' and (target_stats['min'] < -50 or target_stats['max'] > 50):
            reasonable_range = False
            print("  ⚠️ Sıcaklık değerleri makul aralık dışında (-50°C - 50°C)")

        elif key == 'dew_point_temperature' and (target_stats['min'] < -50 or target_stats['max'] > 50):
            reasonable_range = False
            print("  ⚠️ Çiy noktası değerleri makul aralık dışında (-50°C - 50°C)")

        elif key == 'surface_air_pressure' and (target_stats['min'] < 500 or target_stats['max'] > 1100):
            reasonable_range = False
            print("  ⚠️ Basınç değerleri makul aralık dışında (500 hPa - 1100 hPa)")

        elif key == 'relative_humidity' and (target_stats['min'] < 0 or target_stats['max'] > 100):
            reasonable_range = False
            print("  ⚠️ Bağıl nem değerleri makul aralık dışında (0% - 100%)")

        elif key == 'water_vapor_mixing_ratio' and (target_stats['min'] < 0 or target_stats['max'] > 50):
            reasonable_range = False
            print("  ⚠️ Su buharı karışım oranı değerleri makul aralık dışında (0 - 50 g/kg)")

        if reasonable_range:
            print("  ✓ Değerler makul aralıkta")

    else:
        if not feature_exists:
            print(f"  ✗ Değişken '{feature_var}' features veri setinde bulunamadı")
        if not target_exists:
            print(f"  ✗ Değişken '{target_var}' targets veri setinde bulunamadı")

# 4. Fiziksel Tutarlılık Testi
print("\n4. FİZİKSEL TUTARLILIK TESTİ")
print("-" * 50)

# August-Roche-Magnus denklem sabitleri
a_pos = 17.368
b_pos = 238.83
c_pos = 6.107  # hPa
a_neg = 17.856
b_neg = 245.52
c_neg = 6.108  # hPa

# Yeterli değişken var mı?
required_vars = ['obs:air_temperature', 'obs:dew_point_temperature',
                 'obs:surface_air_pressure', 'obs:relative_humidity',
                 'obs:water_vapor_mixing_ratio']

all_vars_exist = all(var in targets for var in required_vars)

if all_vars_exist:
    # Veri örneklemi seçimi
    sample_size = min(1000, len(targets.station) * len(targets.t))

    # Rastgele örneklem seçimi için düzleştirilmiş indeks hesapla
    station_indices = np.random.choice(len(targets.station), size=sample_size, replace=True)
    t_indices = np.random.choice(len(targets.t), size=sample_size, replace=True)
    time_indices = np.random.choice(len(targets.forecast_reference_time), size=sample_size, replace=True)

    # Değişkenleri seç
    T = np.array([targets['obs:air_temperature'].values[s, ti, tt]
                  for s, ti, tt in zip(station_indices, time_indices, t_indices)])
    Td = np.array([targets['obs:dew_point_temperature'].values[s, ti, tt]
                   for s, ti, tt in zip(station_indices, time_indices, t_indices)])
    P = np.array([targets['obs:surface_air_pressure'].values[s, ti, tt]
                  for s, ti, tt in zip(station_indices, time_indices, t_indices)])
    RH = np.array([targets['obs:relative_humidity'].values[s, ti, tt]
                   for s, ti, tt in zip(station_indices, time_indices, t_indices)])
    r = np.array([targets['obs:water_vapor_mixing_ratio'].values[s, ti, tt]
                  for s, ti, tt in zip(station_indices, time_indices, t_indices)])

    # NaN değerleri filtrele
    valid_idx = ~np.isnan(T) & ~np.isnan(Td) & ~np.isnan(P) & ~np.isnan(RH) & ~np.isnan(r)
    T = T[valid_idx]
    Td = Td[valid_idx]
    P = P[valid_idx]
    RH = RH[valid_idx]
    r = r[valid_idx]

    if len(T) > 0:
        # 1. Kontrol: Çiy noktası sıcaklıktan yüksek olmamalı
        td_valid = np.all(Td <= T)
        print(f"1. Çiy noktası sıcaklıktan yüksek değil: {'✓ EVET' if td_valid else '✗ HAYIR'}")
        if not td_valid:
            invalid_count = np.sum(Td > T)
            print(f"   ⚠️ {invalid_count} örnekte çiy noktası sıcaklıktan yüksek")

        # 2. Kontrol: Bağıl nem 0-100 arasında olmalı
        rh_valid = np.all((RH >= 0) & (RH <= 100))
        print(f"2. Bağıl nem 0-100 arasında: {'✓ EVET' if rh_valid else '✗ HAYIR'}")
        if not rh_valid:
            invalid_low = np.sum(RH < 0)
            invalid_high = np.sum(RH > 100)
            print(f"   ⚠️ {invalid_low} örnekte bağıl nem 0'dan küçük, {invalid_high} örnekte 100'den büyük")

        # 3. Kontrol: Bağıl nem, T ve Td ile tutarlı olmalı
        # RH hesabı: RH = 100 * exp((a*Td)/(b+Td)) / exp((a*T)/(b+T))
        a = np.where(T >= 0, a_pos, a_neg)
        b = np.where(T >= 0, b_pos, b_neg)

        # Hesaplanan bağıl nem
        calculated_RH = 100 * np.exp((a * Td) / (b + Td)) / np.exp((a * T) / (b + T))
        rh_diff = np.abs(calculated_RH - RH)
        rh_consistency = np.mean(rh_diff)

        print(f"3. Bağıl nem tutarlılığı (T ve Td ile hesaplanan RH - verilen RH farkı):")
        print(f"   Ortalama fark: {rh_consistency:.2f}%")
        print(f"   Maksimum fark: {np.max(rh_diff):.2f}%")
        print(f"   Tutarlılık: {'✓ İYİ' if rh_consistency < 5 else '⚠️ ORTA' if rh_consistency < 10 else '✗ ZAYIF'}")

        # 4. Kontrol: Su buharı karışım oranı P ve Td ile tutarlı olmalı
        # r = 622 * e / (P - e), e = c * exp(a*Td/(b+Td))
        c = np.where(T >= 0, c_pos, c_neg)
        e = c * np.exp((a * Td) / (b + Td))
        calculated_r = 622.0 * e / (P - e + 1e-6)  # Sayısal kararlılık için epsilon
        r_diff = np.abs(calculated_r - r)
        r_consistency = np.mean(r_diff)

        print(f"4. Su buharı karışım oranı tutarlılığı (P ve Td ile hesaplanan r - verilen r farkı):")
        print(f"   Ortalama fark: {r_consistency:.2f} g/kg")
        print(f"   Maksimum fark: {np.max(r_diff):.2f} g/kg")
        print(f"   Tutarlılık: {'✓ İYİ' if r_consistency < 0.5 else '⚠️ ORTA' if r_consistency < 1 else '✗ ZAYIF'}")

        # 5. Karşılaştırma için scatter plot
        plt.figure(figsize=(15, 10))

        # RH karşılaştırma
        plt.subplot(2, 2, 1)
        plt.scatter(RH, calculated_RH, alpha=0.5, s=10)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Verilen Bağıl Nem (%)')
        plt.ylabel('Hesaplanan Bağıl Nem (%)')
        plt.title('Bağıl Nem Tutarlılığı')
        plt.grid(True)

        # r karşılaştırma
        plt.subplot(2, 2, 2)
        plt.scatter(r, calculated_r, alpha=0.5, s=10)
        max_val = max(np.max(r), np.max(calculated_r))
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Verilen Karışım Oranı (g/kg)')
        plt.ylabel('Hesaplanan Karışım Oranı (g/kg)')
        plt.title('Su Buharı Karışım Oranı Tutarlılığı')
        plt.grid(True)

        # T ve Td karşılaştırma
        plt.subplot(2, 2, 3)
        plt.scatter(T, Td, alpha=0.5, s=10)
        min_val = min(np.min(T), np.min(Td))
        max_val = max(np.max(T), np.max(Td))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Sıcaklık (°C)')
        plt.ylabel('Çiy Noktası Sıcaklığı (°C)')
        plt.title('Sıcaklık ve Çiy Noktası İlişkisi')
        plt.grid(True)

        # Features vs Targets karşılaştırma
        if 'coe:air_temperature_ensavg' in features:
            plt.subplot(2, 2, 4)
            f_sample = []
            t_sample = []

            for s, ti, tt in zip(station_indices[:100], time_indices[:100], t_indices[:100]):
                try:
                    f_val = features['coe:air_temperature_ensavg'].values[s, ti, tt]
                    t_val = targets['obs:air_temperature'].values[s, ti, tt]
                    if not np.isnan(f_val) and not np.isnan(t_val):
                        f_sample.append(f_val)
                        t_sample.append(t_val)
                except IndexError:
                    continue

            if len(f_sample) > 0:
                plt.scatter(f_sample, t_sample, alpha=0.5, s=10)
                min_val = min(np.min(f_sample), np.min(t_sample))
                max_val = max(np.max(f_sample), np.max(t_sample))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.xlabel('NWP Sıcaklık Tahmini (°C)')
                plt.ylabel('Gözlem Sıcaklığı (°C)')
                plt.title('NWP Tahmini vs Gözlem')
                plt.grid(True)

        plt.tight_layout()
        plt.savefig('veri_tutarlilik_analizi.png', dpi=300, bbox_inches='tight')
        print(f"\nGrafik analizi 'veri_tutarlilik_analizi.png' olarak kaydedildi")
    else:
        print("✗ Yeterli geçerli veri yok")
else:
    print("✗ Gerekli değişkenlerden bazıları eksik:")
    for var in required_vars:
        if var not in targets:
            print(f"  - '{var}' eksik")

# 5. Zamanlama Tutarlılığı
print("\n5. ZAMANLAMA TUTARLILIĞI")
print("-" * 50)

if 'forecast_reference_time' in features.dims and 'forecast_reference_time' in targets.dims:
    f_times = features.forecast_reference_time.values
    t_times = targets.forecast_reference_time.values

    # Zaman aralıkları eşleşiyor mu?
    time_match = np.array_equal(f_times, t_times)

    print(f"Zaman aralıkları eşleşiyor mu? {'✓ EVET' if time_match else '✗ HAYIR'}")

    if not time_match:
        f_start, f_end = f_times[0], f_times[-1]
        t_start, t_end = t_times[0], t_times[-1]

        print(f"Features zaman aralığı: {f_start} - {f_end}")
        print(f"Targets zaman aralığı: {t_start} - {t_end}")

        # Farklılıklar
        only_in_features = len(set(f_times) - set(t_times))
        only_in_targets = len(set(t_times) - set(f_times))

        print(f"Sadece features'da bulunan zaman sayısı: {only_in_features}")
        print(f"Sadece targets'da bulunan zaman sayısı: {only_in_targets}")
else:
    missing = []
    if 'forecast_reference_time' not in features.dims:
        missing.append("Features")
    if 'forecast_reference_time' not in targets.dims:
        missing.append("Targets")
    print(f"'forecast_reference_time' boyutu şurada eksik: {', '.join(missing)}")

# GENEL SONUÇ
print("\n" + "=" * 80)
print("GENEL SONUÇ")
print("=" * 80)

all_tests_passed = all_dims_match and all_coords_match and all_vars_exist

# Final özet
if all_tests_passed:
    print("✅ Tüm temel testler başarılı. Veri setleri uyumlu görünüyor.")
else:
    print("⚠️ Bazı testler başarısız. Veri setleri arasında uyumsuzluklar var.")

print("\n🔍 Detaylı analiz için yukarıdaki test sonuçlarını inceleyiniz.")
print("📊 Grafiksel analiz için 'veri_tutarlilik_analizi.png' dosyasına bakınız.")