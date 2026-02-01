# Compression-lab
## Запуск
Выполнить: 
* compress.py 
* train.py
* inference.py

## Файлы
* Сжатая модель: https://drive.google.com/drive/folders/1ffsxkTv97vOrzYbz2jEHj-wkn-BBIkQG?usp=sharing
* Peft-данные дообучения для сжатой модели: https://drive.google.com/drive/folders/1B4d1BtgSvomybpX51PJpFhVPXH849D7d?usp=sharing

## Результаты:

| Модель  | Вес | Точность (MLLU, 10 вопросов на каждый топик) |
| ------------- | ------------- |------------- |
| Оригинальная Qwen3-8B | 15,25 гб | 0.673 |
| Сжатая Qwen3-8B-4bit  | 5,55 гб | 0.657 |
| Сжатая и дообученная Qwen3-8B-4bit final | 5,71 гб |  |
