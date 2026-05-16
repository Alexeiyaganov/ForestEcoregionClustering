# Forest Ecoregion Clustering

> Воспроизведение методологии кластеризации лесных ландшафтов России  
> По статье: Kharitonova et al. (2025) — Doklady Earth Sciences

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alexeiyaganov/ForestEcoregionClustering/blob/main/colab_setup.ipynb)

---

## О проекте

Этот проект воспроизводит подход к районированию лесных территорий, описанный в статье:

> Kharitonova T.I., Krinitskiy M.A., Rezvov V.Yu., Maksakov A.I., Olchev A.V., Gulev S.K.  
> **Regionalization of Forest Landscapes in Russia to Optimize Regional Modeling of Greenhouse Gas Fluxes**  
> *Doklady Earth Sciences*, 2025, Vol. 520, No. 1, pp. 333–339  
> DOI: [10.1134/S1028334X24604346](https://doi.org/10.1134/S1028334X24604346)

**Что сделано:**
- Воспроизведена логика многомерной кластеризации (12 переменных)
- Данные приведены к сетке ~250 м
- Выделены экорегионы для европейской части России
- Построены карты и метрики качества

---

## Быстрый старт в Google Colab

### Шаг 1. Открыть ноутбук

Нажмите на зелёную кнопку **Open In Colab** выше.

Или вручную:
1. Перейдите на https://colab.research.google.com/
2. Выберите **GitHub** вкладку
3. Вставьте ссылку: `https://github.com/Alexeiyaganov/ForestEcoregionClustering`
4. Выберите `colab_setup.ipynb`

### Шаг 2. Запустить все ячейки

В меню Colab: **Runtime → Run all** (или нажмите `Ctrl+F9`)

### Шаг 3. Дождаться выполнения

Процесс занимает 2-3 минуты. Вы увидите:
- Установку библиотек
- Генерацию данных
- Кластеризацию
- Построение графиков

### Шаг 4. Скачать результаты

В последней ячейке ноутбука нажмите на ссылки для скачивания:
- `ecoregion_maps.png` — карта экорегионов
- `elbow_method.png` — график выбора числа кластеров
- `ecoregions_interactive.html` — интерактивная карта
- `results_summary.txt` — метрики качества

---

## Локальный запуск

```bash
# Клонировать репозиторий
git clone https://github.com/Alexeiyaganov/ForestEcoregionClustering.git
cd ForestEcoregionClustering

# Установить зависимости
pip install -r requirements.txt

# Запустить скрипт
python run_clustering.py