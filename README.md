# Технологический стек
python / react / blender

# Содержимое

Здесь существуют следующие папки:

1) datasets - папка с датасетами, которые использовались для обучения нейронной сети
2) Neuro - папка с нейронными сетями
3) dina_prog - папка с алгоритмом динамического программирования, который позволяет быстро рисовать "обводку" растровых изображений

## Реализованный функционал
1) Поиск с помощью нейронных сетей ВЛ (Воздушных линий) и ДКР (Древесно кустарниковой растительности) - находятся в папке Neuro
2) С помощью алгоритмов динамического программирования рисования ОЗ (Охранных зон) для ВЛ и прогнозного на 5 лет уровня роста растительности для ДКР - находятся в папке dina_prog
3) Подсчет площади ДКР, находящихся в ОЗ и возможность подсчета потенциальной площади для ДКР на срок 5 лет, находящихся в ОЗ
4) Выгрузка подсчитанных площадей ДКР для дальнейшего анализа
5) Возмонжость выбора участка ВЛ для оценки
6) Хранение исторических данных по различным ВЛ

## Killer-фичи (Особенности)
1) При обучении нейронной сети использована данные сгенерированные компьютерной графикой (Blender), которые повзолили точно определить высоту ДКР
2) Датасет составлялся с учетом панарамныйх снимков, соотвествтенно высота ДКР определяется с высокой точностью
3) Использование динамического программирования для увеличения точности (идеальный алгоритм для "Задачи береговой линии" - схожа с поиском ДКР)
4) По совокупности это одна из самых точных моделей для определения высоты ДКР

# Тизер

Мы представляем автоматизированную систему расчета площадей и высот растительности в охранной зоне воздушной линии электропередачи "КиберЛес"».

Данная платформа обрабатывает снимки из космоса и, используя нейронные сети, детектирует на них линии электропередач, растительность в их окрестностях, может предсказать их высоту, а также площадь охранной зоны. Высокое качество моделей детекции и сегментации достигнуто использованием технологий компьютерной графики из системы Blender для обогащения набора данных. Набор сцен компьютерной графики позволяет воссоздавать любые растительные зоны с точно расcчитаными параметрами без выезда специалистов на место и дорогостоящих замеров. Помимо этого, можно рассчитать площадь охранной зоны в текущий момент и сделать прогноз на следующие 5 лет с помощью динамического программирования.

Стек решения: python, react, blender

Уникальность: высокая точность определения линий электропередач, возможность получать предсказания по росту растительности и контролировать подрядчиков

# Требования

Для запуска решения требуется установленные python не ниже версии 3.7, утилита pip, совместимая с установленной версией python. Работа на других версиях не проверялась и не может быть гарантирована.
