"""
Forest Ecoregion Clustering
Reproduction of Kharitonova et al. (2025) methodology

Usage:
    python run_clustering.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import folium
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')


def generate_synthetic_data(lat_min=45, lat_max=70, lon_min=30, lon_max=60, resolution=100):
    """Generate synthetic data with 12 variables for European Russia"""
    
    print("   Генерация координатной сетки...")
    lat = np.linspace(lat_min, lat_max, resolution)
    lon = np.linspace(lon_min, lon_max, resolution)
    LON, LAT = np.meshgrid(lon, lat)
    
    np.random.seed(42)
    
    # Climate (4 variables)
    print("   Генерация климатических переменных...")
    temp_annual = 10 - (LAT - 45) * 0.3 + np.random.normal(0, 1, LON.shape)
    precip_annual = 600 - np.abs(LON - 45) * 3 + np.random.normal(0, 30, LON.shape)
    temp_summer = 18 - (LAT - 45) * 0.25 + np.random.normal(0, 1, LON.shape)
    temp_winter = -10 - (LAT - 45) * 0.35 + np.random.normal(0, 1.5, LON.shape)
    
    # Topography
    print("   Генерация переменных рельефа...")
    elevation = np.maximum(0, 500 * np.sin(LAT/20) * np.cos(LON/30) + np.random.normal(0, 50, LON.shape))
    slope = np.abs(np.gradient(elevation)[0]) * 10 + np.abs(np.gradient(elevation)[1]) * 10
    aspect = np.arctan2(np.gradient(elevation)[1], np.gradient(elevation)[0]) * 180 / np.pi
    
    # Soil
    print("   Генерация почвенных переменных...")
    soil_carbon = 50 + elevation/100 + np.random.normal(0, 10, LON.shape)
    soil_sand = 30 + (LAT - 45) * 0.5 + np.random.normal(0, 5, LON.shape)
    soil_moisture = 40 - slope/10 + np.random.normal(0, 8, LON.shape)
    
    # Vegetation
    print("   Генерация переменных растительности...")
    ndvi = 0.6 - (LAT - 45) * 0.015 + np.random.normal(0, 0.05, LON.shape)
    tree_cover = 80 - (LAT - 45) * 1.5 + np.random.normal(0, 5, LON.shape)
    
    # Assemble
    print("   Сборка DataFrame...")
    data = {
        'temp_annual': temp_annual.flatten(),
        'precip_annual': precip_annual.flatten(),
        'temp_summer': temp_summer.flatten(),
        'temp_winter': temp_winter.flatten(),
        'elevation': elevation.flatten(),
        'slope': slope.flatten(),
        'aspect': aspect.flatten(),
        'soil_carbon': soil_carbon.flatten(),
        'soil_sand': soil_sand.flatten(),
        'soil_moisture': soil_moisture.flatten(),
        'ndvi': ndvi.flatten(),
        'tree_cover': tree_cover.flatten(),
        'lat': LAT.flatten(),
        'lon': LON.flatten()
    }
    
    df = pd.DataFrame(data)
    
    # Filter outliers
    print("   Фильтрация выбросов...")
    df = df[(df['temp_annual'] > -15) & (df['temp_annual'] < 15)]
    df = df[(df['precip_annual'] > 200) & (df['precip_annual'] < 1200)]
    df = df[(df['elevation'] >= 0) & (df['elevation'] < 1000)]
    
    return df


def run_clustering(df, n_clusters=18):
    """Run KMeans clustering on 12 variables"""
    
    feature_cols = [
        'temp_annual', 'precip_annual', 'temp_summer', 'temp_winter',
        'elevation', 'slope', 'aspect',
        'soil_carbon', 'soil_sand', 'soil_moisture',
        'ndvi', 'tree_cover'
    ]
    
    print("   Стандартизация признаков...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    print(f"   Запуск KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['ecoregion'] = kmeans.fit_predict(X_scaled)
    
    return df, X_scaled, feature_cols


def evaluate_clustering(df, X_scaled):
    """Calculate quality metrics"""
    
    print("   Вычисление силуэт-коэффициента...")
    sil = silhouette_score(X_scaled, df['ecoregion'])
    
    print("   PCA анализ...")
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_
    
    print("   Сравнение с экспертными зонами...")
    df['latitudinal_zone'] = pd.cut(df['lat'], bins=[0, 50, 60, 90], labels=[0, 1, 2])
    ari = adjusted_rand_score(df['latitudinal_zone'], df['ecoregion'])
    
    return {
        'silhouette': sil,
        'explained_variance_pc1': explained_var[0],
        'explained_variance_pc2': explained_var[1],
        'explained_variance_pc3': explained_var[2],
        'cumulative_variance': explained_var[:3].sum(),
        'adjusted_rand_index': ari,
        'n_clusters': df['ecoregion'].nunique(),
        'n_samples': len(df)
    }


def plot_elbow_method(df, feature_cols, output_dir='results'):
    """Generate elbow plot for optimal k selection"""
    
    print("   Стандартизация для elbow-метода...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    print("   Вычисление инерции для k=2..29...")
    inertias = []
    k_range = range(2, 30)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    print("   Построение графика...")
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, 'o-', color='#2c3e50', linewidth=2)
    plt.axvline(x=18, color='#e74c3c', linestyle='--', label='k = 18 (selected)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Ecoregions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/elbow_method.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_ecoregion_maps(df, output_dir='results'):
    """Generate side-by-side comparison maps"""
    
    print("   Построение карты экорегионов...")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ecoregions
    sc1 = ax[0].scatter(df['lon'], df['lat'], c=df['ecoregion'], cmap='tab20', s=5, alpha=0.7)
    ax[0].set_xlabel('Longitude (°E)')
    ax[0].set_ylabel('Latitude (°N)')
    ax[0].set_title(f'Ecoregions (k={df["ecoregion"].nunique()})')
    ax[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax[0], label='Ecoregion ID')
    
    # Latitudinal zones (expert proxy)
    sc2 = ax[1].scatter(df['lon'], df['lat'], c=df['latitudinal_zone'], cmap='viridis', s=5, alpha=0.7)
    ax[1].set_xlabel('Longitude (°E)')
    ax[1].set_ylabel('Latitude (°N)')
    ax[1].set_title('Expert reference: latitudinal zones')
    ax[1].grid(True, alpha=0.3)
    cbar = plt.colorbar(sc2, ax=ax[1], ticks=[0, 1, 2])
    cbar.set_ticklabels(['South (<50°N)', 'Central (50-60°N)', 'North (>60°N)'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ecoregion_maps.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_interactive_map(df, output_dir='results'):
    """Create interactive Folium map"""
    
    print("   Создание интерактивной карты...")
    n_clusters = df['ecoregion'].nunique()
    sample_df = df.sample(min(1000, len(df)))
    
    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    colors_hex = ['#' + ''.join(f'{int(c*255):02x}' for c in color[:3]) for color in colors]
    
    m = folium.Map(location=[57.5, 45], zoom_start=5, tiles='CartoDB positron')
    
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            popup=f"Ecoregion: {row['ecoregion']}<br>Temperature: {row['temp_annual']:.1f}°C<br>Precipitation: {row['precip_annual']:.0f} mm",
            color=colors_hex[int(row['ecoregion']) % len(colors_hex)],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    m.save(f'{output_dir}/ecoregions_interactive.html')


def save_summary(metrics, output_dir='results'):
    """Save text summary"""
    
    print("   Сохранение текстового отчёта...")
    with open(f'{output_dir}/results_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Forest Ecoregion Clustering - Results Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Number of ecoregions: {metrics['n_clusters']}\n")
        f.write(f"Number of samples: {metrics['n_samples']}\n\n")
        f.write("-" * 40 + "\n")
        f.write("QUALITY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Silhouette score: {metrics['silhouette']:.4f}\n")
        f.write(f"  (range: -1 to 1, >0 indicates some structure)\n\n")
        f.write("PCA explained variance:\n")
        f.write(f"  PC1: {metrics['explained_variance_pc1']:.4f} ({metrics['explained_variance_pc1']*100:.1f}%)\n")
        f.write(f"  PC2: {metrics['explained_variance_pc2']:.4f} ({metrics['explained_variance_pc2']*100:.1f}%)\n")
        f.write(f"  PC3: {metrics['explained_variance_pc3']:.4f} ({metrics['explained_variance_pc3']*100:.1f}%)\n")
        f.write(f"  Cumulative (PC1-3): {metrics['cumulative_variance']:.4f} ({metrics['cumulative_variance']*100:.1f}%)\n\n")
        f.write(f"Adjusted Rand Index (vs latitudinal zones): {metrics['adjusted_rand_index']:.4f}\n")
        f.write(f"  (1.0 = perfect agreement, 0 = random)\n\n")
        f.write("-" * 40 + "\n")
        f.write("REFERENCE\n")
        f.write("-" * 40 + "\n")
        f.write("Kharitonova T.I., Krinitskiy M.A., Rezvov V.Yu., Maksakov A.I., Olchev A.V., Gulev S.K.\n")
        f.write("Regionalization of Forest Landscapes in Russia to Optimize Regional Modeling\n")
        f.write("of Greenhouse Gas Fluxes. Doklady Earth Sciences, 2025, Vol. 520, No. 1, pp. 333-339\n")
        f.write("DOI: 10.1134/S1028334X24604346\n")


def main():
    """Main execution pipeline"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("=" * 60)
    print("Forest Ecoregion Clustering")
    print("Reproduction of Kharitonova et al. (2025)")
    print("=" * 60)
    
    # Generate data
    print("\n[1/7] Генерация синтетических данных...")
    df = generate_synthetic_data()
    print(f"      ✓ Сгенерировано {len(df)} точек")
    
    # Run clustering
    print("\n[2/7] Запуск кластеризации...")
    df, X_scaled, feature_cols = run_clustering(df, n_clusters=18)
    print(f"      ✓ Выделено {df['ecoregion'].nunique()} экорегионов")
    
    # Evaluate
    print("\n[3/7] Вычисление метрик качества...")
    metrics = evaluate_clustering(df, X_scaled)
    print(f"      ✓ Silhouette: {metrics['silhouette']:.4f}")
    print(f"      ✓ ARI: {metrics['adjusted_rand_index']:.4f}")
    
    # Generate plots
    print("\n[4/7] Построение elbow-графика...")
    plot_elbow_method(df, feature_cols)
    print("      ✓ Сохранён: elbow_method.png")
    
    print("\n[5/7] Построение карт...")
    plot_ecoregion_maps(df)
    print("      ✓ Сохранена: ecoregion_maps.png")
    
    # Create interactive map
    print("\n[6/7] Создание интерактивной карты...")
    create_interactive_map(df)
    print("      ✓ Сохранена: ecoregions_interactive.html")
    
    # Save summary
    print("\n[7/7] Сохранение результатов...")
    save_summary(metrics)
    print("      ✓ Сохранён: results_summary.txt")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   • Экорегионов: {metrics['n_clusters']}")
    print(f"   • Силуэт: {metrics['silhouette']:.4f}")
    print(f"   • Объяснённая дисперсия: {metrics['cumulative_variance']*100:.1f}%")
    print(f"   • ARI: {metrics['adjusted_rand_index']:.4f}")
    print("\n📁 Файлы сохранены в папку ./results/")
    print("   - elbow_method.png")
    print("   - ecoregion_maps.png")
    print("   - ecoregions_interactive.html")
    print("   - results_summary.txt")


if __name__ == "__main__":
    main()