# Хакатон "Определи своё место на Физтехе"

Данный репозиторий содержит базовое решение и вспомогательные скрипты для задачи "Распознай своё место на Физтехе" хакатона образовательного форума по искусственному интеллекту, математике и физике 2024.

Презентация задачи доступна по [ссылке](https://docs.google.com/presentation/d/1RfRXQ9x9w3P65KSUNuQcB3zTiU-aLzL4kDCBKoBMHq4/edit?usp=sharing).

## Датасеты

### ITLP-Campus

Датасет записан на робототехнической платформе Husky на территории кампуса МФТИ и состоит из 5 треков, записанных в разное время суток (день/сумерки/ночь) и разные времена года (зима/весна).

Для обучения предлагается использовать зимние треки (день/сумерки/ночь), а весенние (день/ночь) будут выданы для финального тестирования.

Данные разделены по трекам, длина одного трека порядка 3 км, каждый трек включает в себя порядка $600$ фреймов. Расстояние между соседними фреймами ~5 м. Каждый фрейм включает в себя:

- RGB изобрадения для 2х камер (front/back)
- Семантические маски для изображений каждой камеры
- Текстовую информацию, попадющую в поле зрение камеры робота
- 6 DoF позу робота

Подробнее про формат данных и структуру датасета см. [itlp_campus_dataset.md](docs/itlp_campus_dataset.md).

Данные можно найти/использовать на следующих ресурсах:
- [Google Drive](https://drive.google.com/file/d/1txG6aPiy5XtxLOxFk0CxFk9Ey_VUsB5T/view?usp=sharing)

sha256: `3a114e802de48eb12644cd51eeeb757d12524bea59d1563e543989939a522013`  public.zip

Видео-демонстрации треков датасета (публичная часть):
- [2023-02-10-08-04-19-twilight](https://drive.google.com/file/d/1GcJ4jBFuT-Cr4MUTuZaqmX7WDgMNLLJ9/view?usp=share_link)
- [2023-02-21-07-28-58-day](https://drive.google.com/file/d/1BbbCDUx6DnWKaCIgaqZ0Vj9A4-D9WD4Q/view?usp=share_link)
- [2023-03-15-13-25-48-night](https://drive.google.com/file/d/1KiBpk1fBE6cF4BGFK0mPOmsonvB4vBtY/view?usp=share_link)

### Oxford RobotCar

Данные можно найти/использовать на следующих ресурсах:
- [Яндекс.Диск](https://disk.yandex.ru/d/0qq9cnrhlzU8Qg)
- [Kaggle](https://www.kaggle.com/datasets/creatorofuniverses/oxfordrobotcar-iprofi-hack-23)
- [Google Drive](https://drive.google.com/file/d/1b2ry0PGa3vnl8gVhEWqRz329y_ekZX9C/view?usp=share_link)

sha256: `25c45eed9ce77a3a4ab9828754fb1945c358c34d67cc241d47ea0c61d236a620` pnvlad_oxford_robotcar.zip

### NCLT

Данные можно найти/использовать на следующих ресурсах:
- [Яндекс.Диск](https://disk.yandex.ru/d/9wjyWKkWXe0vDQ)
- [Kaggle](https://www.kaggle.com/datasets/creatorofuniverses/nclt-iprofi-hack-23)
- [Google Drive](https://drive.google.com/file/d/192lAPesNgIwbTe9GNqRH0cr9W8Mm16Dv/view?usp=share_link)

sha256: `6ca5dc27d4928b1cbe6c1959b87a539f1dd9bc1764220c53b6d5e406e8cef310` NCLT_preprocessed_small.zip

#### Ручная установка

Код базового решения разрабатывался и тестировался на Ubuntu 20.04 с CUDA 11.6, PyTorch 1.13.1 и MinkowskiEngine 0.5.4

1. Установите PyTorch ~= 1.13.1, воспользовавшись [официальными инструкциями](https://pytorch.org/get-started/previous-versions/)
2. Установите MinkowskiEngine (https://github.com/NVIDIA/MinkowskiEngine):
   ```bash
   # необходимые библиотеки
   sudo apt install build-essential python3-dev libopenblas-dev
   pip install ninja

   # Библиотека MinkowskiEngine должна собираться из исходников с github:
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                          --config-settings="--install-option=--force_cuda" \
                          --config-settings="--install-option=--blas=openblas"
   ```
   Для работы с PyTorch 2.0+ и CUDA 12.X установите MinkowskiEngine из данного репозитория:
   ```bash
   pip install -U git+https://github.com/richlukich/MinkowskiEngine -v --no-deps \
                          --config-settings="--install-option=--force_cuda" \
                          --config-settings="--install-option=--blas=openblas"
   ```

3. Склонируйте и установите [Open Place Recognition](https://github.com/alexmelekhin/OpenPlaceRecognition):
   ```bash
   git clone https://github.com/alexmelekhin/OpenPlaceRecognition
   cd OpenPlaceRecognition
   pip install -e .  # флаг -e необходим для возможности редактировать код уже установленной библиотеки
   ```

После этого вам станет доступен импорт кода из библиотеки `opr`:
```python
from opr.models.place_recognition.base import ImageModel
from opr.modules.feature_extractors import ResNet18FPNFeatureExtractor
from opr.modules import Add, GeM

feature_extractor = ResNet18FPNFeatureExtractor(
    in_channels=3,
    lateral_dim=256,
    fh_num_bottom_up=4,
    fh_num_top_down=0,
    pretrained=True,
)
pooling = GeM()
descriptor_fusion_module = Add()

model = ImageModel(
    backbone=feature_extractor,
    head=pooling,
    fusion=descriptor_fusion_module,
)

```

### Запуск

Демонстрация работы с базовым кодом приведена в ноутбуке [iprofi_baseline_demo.ipynb](./iprofi_baseline_demo.ipynb).

Презентация с семинара-демонстрации доступна по [ссылке](https://docs.google.com/presentation/d/172mSHMW-wVEzzRbEXPIDY14eeKzflan2FSRXjZTRDXM/edit?usp=sharing).

### Визуализация

Для визуализации датасета можно воспользоваться скриптом `scripts/make_video.py` .