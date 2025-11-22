# EDLM VLM Profiling

Проект для профилирования Vision-Language моделей (VLM) с измерением энергопотребления, производительности и качества генерации на различных датасетах.

Выполнили:
- Евгений Здоронок
- Илья Тамбовцев

[Ссылка на презентацию](https://docs.google.com/presentation/d/1HR68rODAJk9CXDNW2O-OZQohVX8OqVOBPVP7vEUzRHo/edit?usp=sharing)

## Функционал

### Поддерживаемые модели

Проект поддерживает профилирование следующих Vision-Language моделей:
- **LLaVA**: `llava-hf/llava-1.5-13b-hf`, `llava-hf/llava-v1.6-mistral-7b-hf`
- **DeepSeek**: `deepseek-ai/deepseek-vl2` (27B)
- **BLIP-2**: `Salesforce/blip2-opt-2.7b`

### Датасеты

Профилирование проводится на трех датасетах:
1. **COCO Captions** - генерация описаний изображений (метрика: ROUGE-L)
2. **TextVQA** - визуальный вопрос-ответ с текстом (метрика: VQA accuracy)
3. **ScienceQA** - научные вопросы с выбором вариантов (метрика: Accuracy)

### Измеряемые метрики

Для каждого эксперимента измеряются:
- **Качество генерации**: ROUGE-L (COCO), VQA Accuracy (TextVQA), Accuracy (ScienceQA)
- **Энергопотребление** (в джоулях) через `pynvml`
- **Утилизация GPU** (%) - средняя загрузка GPU во время inference
- **Утилизация памяти GPU** (%) - средний процент использования памяти
- **Latency** (мс) - время выполнения батчей
- **TFLOPS** - оценка вычислительных операций
- **Количество сгенерированных токенов**

### Параметры профилирования

Проект позволяет варьировать:
- **Размер изображения**: 224×224, 336×336, 448×448, 512×512 пикселей
- **Длина системного промпта**: 10, 50, 100, 200 слов
- **Размер батча**: 1, 4, 16, 64, 320 примеров
- **Интервал семплирования GPU**: настраивается через переменную окружения

## Установка

### Требования
- Менеджер окружений uv
- CUDA 12.4+ (или адаптируйте версию в `pyproject.toml`)
- Linux (протестировано на Ubuntu)
- Проект выполнялся на сервере с A100 и 400Gb оперативной памяти.

### Установка зависимостей

Проект использует [uv](https://github.com/astral-sh/uv) для управления зависимостями:

```bash
# Установка uv (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установка зависимостей
uv sync

# Активация окружения
source .venv/bin/activate
```

Если используете другую версию CUDA, отредактируйте `pyproject.toml`:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }  # для CUDA 12.8
torchvision = { index = "pytorch-cu128" }
```

## Подготовка данных

### Экспорт датасетов

Проект включает утилиты для экспорта данных из популярных датасетов:

```python
from vmp.data.export import export_samples

# Экспорт 300 примеров из каждого датасета
export_samples(
    dataset_name="coco",  # или "textvqa", "scienceqa"
    num_samples=300,
    output_dir="data/coco"
)
```

Также можно использовать ноутбук `notebooks/export_300_samples.ipynb` для интерактивного экспорта.

Структура данных должна быть следующей:
```
data/
├── coco/
│   └── samples_300.pkl
├── textvqa/
│   └── samples_300.pkl
└── scienceqa/
    └── samples_300.pkl
```

### Формат промптов

Промпты для каждого датасета находятся в директории `prompts/`:
- `prompts/coco.yaml` - промпты для генерации описаний
- `prompts/textvqa.yaml` - промпты для TextVQA
- `prompts/scienceqa.yaml` - промпты для ScienceQA

Каждый файл содержит 4 варианта системных промптов разной длины: `system_prompt_10`, `system_prompt_50`, `system_prompt_100`, `system_prompt_200`.

## Воспроизведение результатов

### Конфигурация окружения

Создайте файл `.env` в корне проекта:

```bash
# Отключение XET для HuggingFace Hub
HF_HUB_DISABLE_XET=1

# Указание GPU для использования (например, только GPU 2)
CUDA_VISIBLE_DEVICES=2

# Интервал семплирования GPU метрик в секундах
# Меньшее значение = более частые замеры (более точно, но больше overhead)
GPU_SAMPLING_INTERVAL_S=0.0001

# Включение профилирования VLLM с PyTorch Profiler
# 1 - включено, 0 - выключено
VLLM_ENABLE_PROFILING=0

# Режим отладки (запуск на подмножестве данных)
# 1 - режим отладки, 0 - полный запуск
DEBUG=0

# Выполнить только один батч (для быстрой проверки)
# 1 - один батч, 0 - все батчи
RUN_ONCE=0
```

### Настройка параметров эксперимента

Отредактируйте параметры в файле `run.py`:

```python
# Выбор моделей для профилирования
MODELS = [
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-13b-hf",
    # раскомментируйте нужные модели
]

# Размеры батчей
BATCH_SIZES = [1, 4, 16, 64, 320]

# Размеры изображений
IMAGE_SIZES = [
    (224, 224),
    (336, 336),
    (448, 448),
    (512, 512),
]

# Длины промптов
PROMPTS = [
    "system_prompt_10",
    "system_prompt_50",
    "system_prompt_100",
    "system_prompt_200",
]
```

### Запуск профилирования

```bash
# Полный запуск
python run.py

# Отладочный режим (быстрая проверка на малой выборке)
DEBUG=1 python run.py

# Запуск с одним батчем
RUN_ONCE=1 python run.py

# Запуск с профилированием VLLM
VLLM_ENABLE_PROFILING=1 python run.py
```

### Результаты

После запуска результаты будут сохранены в директории `results/`:

```
results/
├── aggregated_metrics.csv       # Агрегированные метрики всех экспериментов
├── logs/                        # Детальные логи по каждому эксперименту
│   └── YYMMDD_HHMMSS_dataset_model_size_prompt_bsN_raw.csv
└── profiling/                   # Профили PyTorch Profiler (если включено)
    └── YYMMDD_HHMMSS/
```

#### Структура aggregated_metrics.csv

| Колонка | Описание |
|---------|----------|
| `dataset` | Название датасета (coco/textvqa/scienceqa) |
| `model` | Название модели |
| `batch_size` | Размер батча |
| `image_width`, `image_height` | Размер изображения |
| `prompt_len` | Длина промпта (10/50/100/200) |
| `metric` | Значение метрики качества |
| `latency_ms_mean` | Средняя латентность батча (мс) |
| `gpu_util_mean` | Средняя утилизация GPU (%) |
| `mem_util_mean` | Средняя утилизация памяти GPU (%) |
| `energy_j` | Общее энергопотребление (джоули) |
| `est_tflops_total` | Оценка общих TFLOPS |
| `gen_tokens_total` | Общее количество сгенерированных токенов |

## Ноутбуки

- `notebooks/export_300_samples.ipynb`: Экспорт 300 примеров из датасетов COCO, TextVQA и ScienceQA в формат для профилирования. Сохраняет данные в `data/{dataset}/samples_300.pkl`.
- `notebooks/profiling_analysis.ipynb`: Основной ноутбук для анализа результатов профилирования:
- `notebooks/eda_profiling.ipynb` - Exploratory Data Analysis результатов профилирования:
- `notebooks/vlm_profiling.ipynb` - Детальное профилирование отдельных VLM моделей:

## Структура проекта

```
edlm-vlm-profiling/
├── data/                    # Данные датасетов
│   ├── coco/
│   ├── textvqa/
│   └── scienceqa/
├── notebooks/               # Jupyter ноутбуки для анализа
│   ├── export_300_samples.ipynb
│   ├── profiling_analysis.ipynb
│   ├── eda_profiling.ipynb
│   └── vlm_profiling.ipynb
├── prompts/                 # YAML файлы с промптами
│   ├── coco.yaml
│   ├── textvqa.yaml
│   └── scienceqa.yaml
├── results/                 # Результаты экспериментов
│   ├── aggregated_metrics.csv
│   ├── logs/
│   └── profiling/
├── vmp/                     # Основной пакет
│   ├── data/               # Модули для работы с данными
│   ├── utils/              # Утилиты (energy, flops, metrics, images)
│   └── results/            # Обработка результатов
├── run.py                   # Основной скрипт профилирования
├── serve_vllm_model.sh      # Скрипт для запуска VLLM сервера
├── pyproject.toml           # Конфигурация проекта и зависимости
├── .env                     # Переменные окружения (создать самостоятельно)
└── README.md                # Этот файл
```

## Дополнительные возможности

### Запуск VLLM сервера

Для работы с моделями через API можно запустить VLLM сервер (не требуется для воспроизведения результатов):

```bash
# Отредактируйте serve_vllm_model.sh под свои нужды
# Укажите путь к кэшу моделей, порт, GPU и т.д.

bash serve_vllm_model.sh
```

Сервер будет доступен по адресу `http://localhost:16322` (или другому порту, указанному в скрипте).

### Отладка

Для быстрой проверки работоспособности используйте отладочный режим:

```bash
DEBUG=1 RUN_ONCE=1 python run.py
```

Это запустит профилирование на:
- Первой модели из списка
- Одном размере изображения (224×224)
- Одной длине промпта (10 слов)
- Последнем размере батча
- Одном датасете
- 20% данных
- Только одном батче
