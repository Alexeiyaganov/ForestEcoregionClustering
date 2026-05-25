# ForestEcoregionClustering

Воспроизведение и развитие методологии районирования лесных ландшафтов России из статьи:

> Kharitonova T.I., Krinitskiy M.A., Rezvov V.Yu., Maksakov A.I., Olchev A.V., Gulev S.K.
> **Regionalization of Forest Landscapes in Russia to Optimize Regional Modeling of Greenhouse Gas Fluxes**
> *Doklady Earth Sciences*, 2025, Vol. 520, No. 1.
> DOI: [10.1134/S1028334X24604346](https://doi.org/10.1134/S1028334X24604346)

## Что это

Исходная статья строит статичное районирование лесных территорий России методом SLIC-кластеризации по 12 переменным (климат, рельеф, почвы, растительность). Все признаки при этом равноправны.

Этот проект предлагает **task-driven** альтернативу: **Weighted Soft K-Means** — алгоритм, который обучает веса признаков так, чтобы районирование максимизировало качество предсказания потоков CO₂/CH₄ внутри каждого экорегиона. Веса оптимизируются через градиентный спуск (Adam), целевая метрика — взвешенный in-cluster R².

Гипотеза: районирование, оптимизированное под конкретную downstream-задачу, даёт лучшее in-cluster R² по сравнению с геометрическим KMeans при тех же данных.

## Структура репозитория

```
ForestEcoregionClustering/
├── configs/
│   └── config.yaml           # все параметры эксперимента
├── data/
│   ├── fluxnet_loader.py     # загрузка станций FLUXNET2015
│   ├── chelsa_loader.py      # климат, рельеф, почвы, NDVI
│   └── pipeline.py           # сборка датасета
├── models/
│   ├── baseline.py           # KMeans baseline
│   └── weighted_clustering.py  # Weighted Soft K-Means (основной метод)
├── eval/
│   └── metrics.py            # silhouette, in-cluster R², ARI
├── viz/
│   └── plots.py              # карты, веса признаков, кривая обучения
├── notebooks/
│   └── experiment.ipynb      # главный ноутбук для Colab
├── results/                  # сюда сохраняются выходные файлы
└── requirements.txt
```

## Данные

| Источник | Что | Как используется |
|----------|-----|-----------------|
| [FLUXNET2015](https://fluxnet.org/data/fluxnet2015-dataset/) | Измерения потоков CO₂/CH₄ на ~212 станциях | Целевая переменная (NEE, gC/m²/yr) |
| [CHELSA v2.1](https://chelsa-climate.org/) | Климат 1 km, 1981–2010 | 5 bioclim-переменных |
| [MERIT DEM](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/) | Рельеф 90 m | высота, уклон, экспозиция |
| [SoilGrids v2](https://soilgrids.org/) | Почвы 250 m | SOC, clay content |
| [MODIS MOD13A3](https://lpdaac.usgs.gov/products/mod13a3v061/) | Растительность | NDVI, tree cover |

Загрузчики данных работают автоматически: сначала пробуют скачать реальные данные через API, при недоступности переключаются на аналитические аппроксимации. Всё кэшируется в `data/raw/`.

## Запуск

### Разработка — GitHub Codespaces

Репозиторий настроен для редактирования в [GitHub Codespaces](https://github.com/features/codespaces). Все изменения в модулях (`data/`, `models/`, `eval/`, `viz/`) делаются здесь, затем `git push`.

Параметры эксперимента меняются только через `configs/config.yaml` — не нужно трогать код модулей.

### Расчёты — Google Colab

1. Открой [Google Colab](https://colab.research.google.com/)
2. Загрузи ноутбук: **File → Open notebook → GitHub** → вставь URL репозитория → выбери `notebooks/experiment.ipynb`
3. Запусти все ячейки по порядку (Ctrl+F9)

Первая ячейка клонирует репозиторий и устанавливает зависимости автоматически:

```python
!git clone https://github.com/Alexeiyaganov/ForestEcoregionClustering.git repo
%cd repo
!pip install -q -r requirements.txt
```

Для GPU: **Runtime → Change runtime type → T4 GPU** (ускоряет обучение weighted clustering).

### Локальный запуск

```bash
git clone https://github.com/Alexeiyaganov/ForestEcoregionClustering.git
cd ForestEcoregionClustering
pip install -r requirements.txt
jupyter notebook notebooks/experiment.ipynb
```

## Метод

### Baseline: KMeans

Воспроизводит логику Kharitonova et al. (2025): стандартизация 12 признаков, KMeans с равными весами. Метрики: silhouette score, in-cluster R², ARI против IGBP-классификации.

### Weighted Soft K-Means (основной вклад)

Взвешенное расстояние от точки $i$ до центра кластера $k$:

$$d(x_i, c_k) = \sum_j w_j (x_{ij} - c_{kj})^2$$

Мягкое назначение через softmax с температурным параметром $\tau$:

$$p(k \mid i) = \frac{\exp(-d(x_i, c_k)/\tau)}{\sum_{k'} \exp(-d(x_i, c_{k'})/\tau)}$$

Целевой функционал — взвешенный in-cluster R² потоков NEE:

$$\mathcal{L} = -\sum_k \frac{\sum_i p(k|i)}{\sum_{k,i} p(k|i)} \cdot R^2_k + \lambda \|w\|^2$$

Веса $w$ обучаются через Adam, $\tau$ убывает по линейному расписанию (annealing от `temperature` до `temperature_min`).

## Конфигурация

Все параметры в `configs/config.yaml`:

```yaml
clustering:
  n_clusters: 12

weighted_clustering:
  n_epochs: 200
  lr: 0.01
  temperature: 1.0
  temperature_min: 0.1

region:
  lat_min: 50.0
  lat_max: 70.0
  lon_min: 28.0
  lon_max: 70.0
```

## Результаты

*В процессе. Будут добавлены после завершения экспериментов.*

## Автор

**Алексей Яганов** — MLOps Engineer / Data Scientist

GitHub: [Alexeiyaganov](https://github.com/Alexeiyaganov) · Email: btls3@yandex.ru

## Ссылки

- Kharitonova et al. (2025) — [DOI: 10.1134/S1028334X24604346](https://doi.org/10.1134/S1028334X24604346)
- FLUXNET2015 dataset — [fluxnet.org](https://fluxnet.org/data/fluxnet2015-dataset/)
- CHELSA climate data — [chelsa-climate.org](https://chelsa-climate.org/)
- Climate Change AI workshops — [climatechange.ai](https://www.climatechange.ai/events)