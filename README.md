# coin_classification
Проект по классификации монет на 30000 классов. Стояла задача по фотографиям аверса и реверса(не важно в каком порядке) найти 10 самых похожих монет. На каждый класс по одной фотографии аверса и реверса.
Представлен только тот код, который писал я.

На вход подавались монеты уже после Object Detection.

Проводили очень много экспериментов, обучали автокодировщик, чтоб потом извлекать фичи из изображения, каскад классификаторов, сиамские сети. Было решено использовать предобученную модель VGG19, получать эмбеддинги изображения на выходе и искать ближайшие изображения по косинусному растоянию. Для того, чтоб не имело значения, в каком порядке подавать аверс и реверс на вход, эмбеддинги, которые мы получаем на выходе VGG19, складываются друг с другом.

**coin_classifier.ipynb** - файл для тестирования

**create_embeddings.py** - создание эмбеддингов

**create_segmentation_model.py** - модель сегментации

**image_downloader.py** - скетч для скачивания тренировочных изображений

**segmentation.ipynb** - обучение сегментации(изображения не стал сюда загружать)
