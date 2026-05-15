import pandas as pd

# =============================================
# LANGKAH 1: Load dataset
# =============================================
print("=" * 40)
print("LOADING DATASET...")
print("=" * 40)

df = pd.read_csv('data/umkm_dataset.csv')

print(f"✓ Dataset berhasil dimuat!")
print(f"✓ Jumlah data: {len(df)} baris")
print(f"✓ Jumlah kolom: {len(df.columns)} kolom")

# =============================================
# LANGKAH 2: Lihat 5 data pertama
# =============================================
print("\n" + "=" * 40)
print("5 DATA PERTAMA:")
print("=" * 40)
print(df.head())

# =============================================
# LANGKAH 3: Info kolom dan tipe data
# =============================================
print("\n" + "=" * 40)
print("INFO KOLOM:")
print("=" * 40)
print(df.info())

# =============================================
# LANGKAH 4: Berapa banyak tiap rekomendasi?
# =============================================
print("\n" + "=" * 40)
print("JUMLAH TIAP REKOMENDASI BISNIS:")
print("=" * 40)
print(df['rekomendasi'].value_counts())

# =============================================
# LANGKAH 5: Berapa banyak tiap modal?
# =============================================
print("\n" + "=" * 40)
print("DISTRIBUSI MODAL:")
print("=" * 40)
print(df['modal'].value_counts())

print("\n✓ Eksplorasi selesai! Data siap diproses.")