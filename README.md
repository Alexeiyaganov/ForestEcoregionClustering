# Forest Ecoregion Clustering

> Воспроизведение методологии кластеризации лесных ландшафтов России  
> По статье: Kharitonova et al. (2025) — Doklady Earth Sciences

**Автор:** Алексей Яганов  
**Email:** btls3@yandex.ru  
**GitHub:** [Alexeiyaganov](https://github.com/Alexeiyaganov)

---

## Что это?

Этот проект воспроизводит подход к районированию лесных территорий, описанный в статье:

> Kharitonova T.I., Krinitskiy M.A., Rezvov V.Yu., Maksakov A.I., Olchev A.V., Gulev S.K.  
> **Regionalization of Forest Landscapes in Russia to Optimize Regional Modeling of Greenhouse Gas Fluxes**  
> *Doklady Earth Sciences*, 2025, Vol. 520, No. 1, pp. 333–339  
> DOI: [10.1134/S1028334X24604346](https://doi.org/10.1134/S1028334X24604346)

**Что сделано:**
- Воспроизведена логика многомерной кластеризации (12 переменных)
- Данные приведены к сетке ~250 м
- Выделены экорегионы для европейской части России (18 штук)
- Построены карты и рассчитаны метрики качества

---

## Как запустить в Google Colab (пошаговая инструкция)

### Шаг 1. Откройте Google Colab

Перейдите по ссылке: https://colab.research.google.com/

### Шаг 2. Создайте новый ноутбук

Нажмите **Файл → Создать новый блокнот** (или `Ctrl + N`)

### Шаг 3. Скопируйте код в первую ячейку

В появившуюся ячейку вставьте этот код:

```python
# Клонируем репозиторий с проектом
!git clone https://github.com/Alexeiyaganov/ForestEcoregionClustering.git

# Переходим в папку проекта
%cd ForestEcoregionClustering

# Устанавливаем необходимые библиотеки
!pip install -q -r requirements.txt

# Запускаем основной скрипт
!python run_clustering.py

Результаты появятся в папке results/

Результаты (синтетические данные)
Метрика	Значение
Число экорегионов	18
Силуэт-коэффициент	~0.10
Объяснённая дисперсия (PC1-3)	~66%
Adjusted Rand Index	~0.12
Эти результаты — baseline на синтетических данных. Низкий силуэт согласуется с наблюдениями авторов для северо-восточной Сибири (SS = –0.01 до 0.03).

Что дальше (предлагаемые улучшения)
Реальные данные — CHELSA (климат), MERIT DEM (рельеф), SoilGrids (почвы), MODIS (NDVI)

Высокое разрешение — 30 м вместо 250 м

Микрорельеф — добавление экспозиции и TWI

SLIC вместо KMeans — учёт пространственной связности

Динамика — кластеризация по годам (2000-2025)

MLOps-пайплайн — DVC + MLflow + CI/CD

Структура репозитория

ForestEcoregionClustering/
├── README.md              # Этот файл
├── requirements.txt       # Зависимости Python
├── run_clustering.py      # Основной скрипт
├── colab_setup.ipynb      # Обёртка для Google Colab
└── results/               # Папка для результатов
Автор
Алексей Яганов

MLOps Engineer / Data Scientist

Мастер спорта по спортивному ориентированию

Email: btls3@yandex.ru

GitHub: Alexeiyaganov