# Morphology Work

Проект содержит статью и практическую реализацию локальных бинарных морфологических операторов на C++ и OpenCV.

Реализованы операции:

- `hitmiss`
- `clean`
- `fill`
- `majority`
- `endpoints`
- `spur`
- `bridge`
- `hbreak`

## Зависимости

Нужны:

- C++17 compiler, например `g++`
- `make`
- OpenCV 4
- `pkg-config`

Проверка OpenCV:

```bash
pkg-config --modversion opencv4
```

## Сборка

```bash
make
```

После сборки исполняемый файл будет здесь:

```bash
build/morphlab
```

Очистка сборки:

```bash
make clean
```

## Запуск экспериментов

```bash
make experiments
```

Эта команда создаёт артефакты в каталоге `results/`:

- `experiment_summary.md` — краткая сводка результатов;
- `exp0_dilation_decomposition_montage.png` — проверка разложения дилатации `3x3` на `1x3` и `3x1`;
- `exp1_hitmiss_montage.png` — визуализация hit-or-miss;
- `exp2_lut_*.csv` — таблицы истинности для 512 окрестностей по всем LUT-операторам;
- `exp3_bridge_montage.png` — результат `bridge`;
- `exp4_hbreak_montage.png` — результат `hbreak`;
- `exp5_spur_montage.png` — результат `endpoints` и `spur`.

## Интерактивный GUI

Запуск:

```bash
make gui
```

То же самое напрямую:

```bash
build/morphlab --gui
```

В окне отображаются три панели:

- `canvas` — исходный бинарный холст;
- панель выбранной операции — результат обработки;
- `diff` — карта изменений.

Управление:

- левая кнопка мыши — рисовать объектные пиксели;
- правая кнопка мыши — стирать пиксели;
- `1`..`8` — выбрать операцию: `clean`, `fill`, `majority`, `endpoints`, `spur`, `bridge`, `hbreak`, `hitmiss`;
- `Space` или `Enter` — применить выбранную операцию;
- `[` и `]` — уменьшить или увеличить число итераций;
- `-` и `+` — уменьшить или увеличить кисть;
- `c` — очистить холст;
- `e` — загрузить встроенный пример с разрывами для `bridge`;
- `u` — перенести результат обратно на холст;
- `s` — сохранить `results/gui_canvas.png`, `results/gui_result.png`, `results/gui_diff.png`;
- `q` или `Esc` — выйти.

GUI использует OpenCV HighGUI, поэтому его нужно запускать в окружении с доступным графическим дисплеем.

## Запуск на своём изображении

Общий вид:

```bash
build/morphlab --input input.png --operation bridge --iterations 1 --output result.png
```

С diff-картой:

```bash
build/morphlab \
  --input input.png \
  --operation bridge \
  --iterations 1 \
  --output result.png \
  --diff diff.png
```

По умолчанию бинаризация выполняется методом Оцу. Чтобы задать ручной порог:

```bash
build/morphlab --input input.png --threshold 127 --operation clean --output clean.png
```

Для `hitmiss` используется встроенный демонстрационный шаблон правого конца горизонтальной линии:

```bash
build/morphlab --input input.png --operation hitmiss --output matches.png
```

## Вывод метрик

После обработки программа печатает в stdout:

```text
pixels_before=... pixels_after=... components8_before=... components8_after=... changed=... added=... removed=...
```

Где:

- `pixels_before`, `pixels_after` — число объектных пикселей;
- `components8_before`, `components8_after` — число 8-связных компонент;
- `changed` — число изменённых пикселей;
- `added` — число добавленных пикселей;
- `removed` — число удалённых пикселей.

## Статья

Основной текст работы находится в:

```text
work.md
```

Статья ссылается на изображения и таблицы из `results/`, поэтому перед просмотром финальной версии стоит выполнить:

```bash
make experiments
```
